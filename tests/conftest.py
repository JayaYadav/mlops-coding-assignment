import pytest
import numpy as np
import io
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def api_client():
    """FastAPI test client for integration testing."""
    return TestClient(app)

@pytest.fixture
def valid_image_bytes():
    """Generates a valid 28x28 grayscale image in bytes."""
    img_arr = np.zeros((28, 28), dtype=np.uint8)
    img = Image.fromarray(img_arr)
    byte_arr = io.BytesIO()
    img.save(byte_arr, format='PNG')
    return byte_arr.getvalue()

@pytest.fixture
def invalid_image_bytes():
    """Generates an invalid image format to trigger validation errors."""
    return b"this is not an image file"