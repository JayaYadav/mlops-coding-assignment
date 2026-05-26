import io
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger("digit-api.predictor")

class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 7 * 7, 128)

    def forward(self, x):
        return self.fc(self.flatten(self.conv(x)))


class FinalClassifier(nn.Module):
    def __init__(self, image_feat_dim=128, metadata_dim=6, num_classes=10):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(image_feat_dim + metadata_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, img_feat, meta_feat):
        return self.fc(torch.cat([img_feat, meta_feat], dim=1))


class MultimodalDigitPredictor:
    def __init__(self, project_root: Path):
        model_path = project_root / "models" / "image_model.pth"
        classifier_path = project_root / "models" / "final_classifier.pth"
        encoder_path = project_root / "models" / "metadata_encoder.joblib"

        try:
            # 1. Load Metadata Encoder & Calculate dimensions dynamically
            self.metadata_encoder = joblib.load(str(encoder_path))
            sample_meta = pd.DataFrame([{"pen_pressure": 1.0, "writer_age": 30, "handedness": "right"}])
            metadata_dim = self.metadata_encoder.transform(sample_meta).shape[1]

            # 2. Load PyTorch Image Encoder
            self.image_model = CNNEncoder()
            self.image_model.load_state_dict(torch.load(str(model_path), map_location="cpu"))
            self.image_model.eval()

            # 3. Load Combined Classifier
            self.final_model = FinalClassifier(metadata_dim=metadata_dim)
            self.final_model.load_state_dict(torch.load(str(classifier_path), map_location="cpu"))
            self.final_model.eval()

            logger.info("All ML model components loaded successfully into memory.")
        except Exception as e:
            logger.critical(f"FAIL-FAST: Missing or corrupt model weights file. Stack: {e}", exc_info=True)
            raise SystemExit("Application aborting: Cannot read local model files.")

    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("L")
            
            # Target sizing check: Enforcing 28x28 if that is what the CNN expects
            if img.size != (28, 28):
                logger.warning(f"Resizing incoming image from {img.size} to (28, 28)")
                img = img.resize((28, 28))

            img_arr = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.tensor(img_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # Shape: (1, 1, 28, 28)
            return img_tensor
        except Exception as e:
            raise ValueError(f"Image preprocessing failed. File corrupted or invalid matrix: {e}")

    def encode_metadata(self, pen_pressure: float, writer_age: int, handedness: str) -> torch.Tensor:
        try:
            meta_df = pd.DataFrame(
                [{"pen_pressure": pen_pressure, "writer_age": writer_age, "handedness": handedness}]
            )
            encoded = self.metadata_encoder.transform(meta_df).astype(np.float32)
            return torch.tensor(encoded)
        except Exception as e:
            raise ValueError(f"Failed transformation within tabular pipeline encoder: {e}")

    def predict(self, image_bytes: bytes, pen_pressure: float, writer_age: int, handedness: str) -> int:
        # Preprocess both components
        img_tensor = self.preprocess_image(image_bytes)
        meta_tensor = self.encode_metadata(pen_pressure, writer_age, handedness)

        # Execute thread-safe inference
        with torch.no_grad():
            img_feat = self.image_model(img_tensor)
            logits = self.final_model(img_feat, meta_tensor)
            prediction = int(torch.argmax(logits, dim=1).item())
        return prediction