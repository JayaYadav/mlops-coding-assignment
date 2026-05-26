import pytest
import numpy as np
import io
from PIL import Image
from pathlib import Path
import torch
from app.predictor import MultimodalDigitPredictor

@pytest.fixture
def predictor():
    PROJECT_ROOT = Path(__file__).parent.parent
    return MultimodalDigitPredictor(project_root=PROJECT_ROOT)

def test_image_preprocessing_resizes_large_images(predictor):
    """Verifies that an incoming 100x100 color image gets scaled down to a 1x1x28x28 tensor."""
    # Create a 100x100 RGB image
    img_arr = np.zeros((100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(img_arr)
    byte_arr = io.BytesIO()
    img.save(byte_arr, format='PNG')
    
    tensor = predictor.preprocess_image(byte_arr.getvalue())
    
    # Assert exact expected tensor shape for PyTorch CNN
    assert tensor.shape == (1, 1, 28, 28)
    assert tensor.dtype == torch.float32

def test_image_preprocessing_fails_with_corrupt_data(predictor):
    """Verifies that corrupt image data throws a clear ValueError."""
    with pytest.raises(ValueError, match="Image preprocessing failed"):
        predictor.preprocess_image(b"corrupt payload strings")