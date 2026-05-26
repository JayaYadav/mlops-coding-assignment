import io
import logging
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("digit-api")

# Get the project root directory (parent of the app directory)
PROJECT_ROOT = Path(__file__).parent.parent


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
        x = self.conv(x)
        x = self.flatten(x)
        return self.fc(x)


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


app = FastAPI(title="MNIST Digit Prediction API")

MODEL_PATH = PROJECT_ROOT / "models" / "image_model.pth"
CLASSIFIER_PATH = PROJECT_ROOT / "models" / "final_classifier.pth"
ENCODER_PATH = PROJECT_ROOT / "models" / "metadata_encoder.joblib"


def load_models():
    try:
        image_model = CNNEncoder()
        image_model.load_state_dict(torch.load(str(MODEL_PATH), map_location="cpu"))
        image_model.eval()

        metadata_encoder = joblib.load(str(ENCODER_PATH))

        # determine metadata dimension from encoder output shape
        sample_meta = pd.DataFrame(
            [{"pen_pressure": 1.0, "writer_age": 30, "handedness": "right"}]
        )
        metadata_dim = metadata_encoder.transform(sample_meta).shape[1]

        final_model = FinalClassifier(metadata_dim=metadata_dim)
        final_model.load_state_dict(torch.load(str(CLASSIFIER_PATH), map_location="cpu"))
        final_model.eval()

        logger.info("Models loaded successfully")
        return image_model, final_model, metadata_encoder
    except Exception as exc:
        logger.error("Failed to load trained files: %s", exc)
        raise


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    img_arr = np.array(img).astype(np.float32) / 255.0
    if img_arr.ndim != 2:
        raise ValueError("Image must be grayscale or convertible to grayscale")
    img_tensor = torch.tensor(img_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return img_tensor


def encode_metadata(metadata_encoder: ColumnTransformer, pen_pressure: float, writer_age: int, handedness: str) -> torch.Tensor:
    meta_df = pd.DataFrame(
        [{"pen_pressure": pen_pressure, "writer_age": writer_age, "handedness": handedness}]
    )
    encoded = metadata_encoder.transform(meta_df).astype(np.float32)
    return torch.tensor(encoded)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(
    image: UploadFile = File(...),
    pen_pressure: float = Form(...),
    writer_age: int = Form(...),
    handedness: str = Form(...),
):
    try:
        image_bytes = image.file.read()
        img_tensor = preprocess_image(image_bytes)
    except Exception as exc:
        logger.exception("Invalid image upload")
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    try:
        image_model, final_model, metadata_encoder = load_models()
        meta_tensor = encode_metadata(metadata_encoder, pen_pressure, writer_age, handedness)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to load model artifacts")

    try:
        with torch.no_grad():
            img_feat = image_model(img_tensor)
            logits = final_model(img_feat, meta_tensor)
            pred = int(torch.argmax(logits, dim=1).item())
        return JSONResponse(content={"predicted_digit": pred})
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")
