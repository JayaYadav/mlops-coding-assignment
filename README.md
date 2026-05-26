# MNIST Digit Prediction API

## Overview

This project serves a handwritten digit prediction model through a FastAPI inference service. The API accepts an image file plus handwriting metadata, validates inputs, and returns a digit prediction.

## Prerequisites

- Python 3.11
- Git
- Docker (for container build and run)
- Optional: virtual environment tooling

## Local set up - The hard way

### Setup Locally

1. Clone the repository

```bash
git clone https://github.com/your-organization/your-repo.git
cd mlops-coding-assignment
```

2. Create and activate a Python virtual environment

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install Python dependencies

```bash
pip install --no-cache-dir -r requirements.txt
```

4. Verify required model artifacts are available

The service expects these files in the repository root:

- `image_model.pth`
- `final_classifier.pth`
- `metadata_encoder.joblib`

These artifacts are included in the repository for the assignment.

### Run the API Locally

Start the application with Uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:

- Health: `http://127.0.0.1:8000/health`
- Prediction: `http://127.0.0.1:8000/predict`

#### Example Request

Use a multipart form request to send the image and metadata:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "image=@sample_images/example.png" \
  -F "pen_pressure=5.0" \
  -F "writer_age=32" \
  -F "handedness=left"
```

Example successful response:

```json
{"predicted_digit": 7}
```

### Test the Project

Install test dependencies and run the test suite.

```bash
pip install pytest
pytest -q
```

If you prefer, run:

```bash
python -m pytest -q
```

## Out of the box execution - The easy way

### Build the Docker Container

Build the image from the project root:

```bash
docker build -t mlops-digit-api .
```

### Run the Docker Container

Expose port `8000` and start the container:

```bash
docker run --rm -p 8000:8000 mlops-digit-api
```

Then use the same `/predict` and `/health` endpoints against `http://127.0.0.1:8000`.
An easier way to test is visiting `http://127.0.0.1:8000/docs` on your browser. Try out the predict endpoint. Upload a sample image from sample_images folder.

## Notes

- The application is served by `uvicorn` using the FastAPI app object in `app/main.py`.
- The API includes basic observability: health checks, request logging, and a latency header (`X-Process-Time-Ms`).
- The `tests/` folder includes automated API coverage with `fastapi.testclient`.
- The best practice is to add trained artifacts to gitignore files and not push them to the remote repo. But since the coding assignment's ask is to run the setup out-of-the-box, the artifacts are being pushed to the remote repo.