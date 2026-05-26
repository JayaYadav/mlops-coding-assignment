# tests/test_integration.py
def test_predict_endpoint_success(api_client, valid_image_bytes):
    """Tests a full successful request cycle through the API."""
    payload = {
        "pen_pressure": 5.0,
        "writer_age": 32,
        "handedness": "left"
    }
    files = {
        "image": ("digit.png", valid_image_bytes, "image/png")
    }
    
    response = api_client.post("/predict", data=payload, files=files)
    
    assert response.status_code == 200
    assert "predicted_digit" in response.json()
    # Check that your observability metric header is present in the response
    assert "X-Process-Time-Ms" in response.headers

def test_predict_endpoint_invalid_metadata(api_client, valid_image_bytes):
    """Tests that bad form fields are safely caught and return a 422 error."""
    payload = {
        "pen_pressure": -10.0,  # Invalid negative pressure
        "writer_age": 32,
        "handedness": "left"
    }
    files = {
        "image": ("digit.png", valid_image_bytes, "image/png")
    }
    
    response = api_client.post("/predict", data=payload, files=files)
    assert response.status_code == 422
    assert response.json()["error"] == "InvalidInputMetadata"

def test_health_endpoint(api_client):
    """Tests the health check route."""
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}