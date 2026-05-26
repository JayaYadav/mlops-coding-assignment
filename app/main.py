import logging
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
import time

from app.schemas import MetadataInput
from app.predictor import MultimodalDigitPredictor

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("digit-api.main")

app = FastAPI(title="MNIST Digit Prediction API")

PROJECT_ROOT = Path(__file__).parent.parent

# Initialize predictor globally once on startup
predictor = MultimodalDigitPredictor(project_root=PROJECT_ROOT)

# Middleware to measure and log performance metrics for each request
@app.middleware("http")
async def add_performance_metrics(request: Request, call_next):
    start_time = time.time()
    
    # Process the incoming request and get the response
    response = await call_next(request)
    
    # Calculate execution duration in milliseconds
    process_time_ms = (time.time() - start_time) * 1000
    
    # Inject custom monitoring headers into the response
    response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}ms"
    
    # Log the basic performance metric
    logger.info(
        f"METRIC: Path={request.url.path} | "
        f"Status={response.status_code} | "
        f"Latency={process_time_ms:.2f}ms"
    )
    
    return response

# Basic health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}

# Main prediction endpoint
@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    pen_pressure: float = Form(...),
    writer_age: int = Form(...),
    handedness: str = Form(...),
):
    # 1. Structural/Input validation layer for metadata
    try:
        validated_meta = MetadataInput.from_form(
            pen_pressure=pen_pressure, writer_age=writer_age, handedness=handedness
        )
    except ValidationError as ve:
        logger.warning(f"Metadata input validation failed: {ve.errors()}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"error": "InvalidInputMetadata", "details": ve.errors()}
        )

    # 2. Extract files
    try:
        image_bytes = await image.read()
        if not image_bytes:
            raise ValueError("Uploaded file payload is completely empty.")
    except Exception as exc:
        logger.error(f"Failed to read image file stream: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Could not process file upload stream securely."
        )

    # 3. Model Pipeline processing layer
    try:
        pred_digit = predictor.predict(
            image_bytes=image_bytes,
            pen_pressure=validated_meta.pen_pressure,
            writer_age=validated_meta.writer_age,
            handedness=validated_meta.handedness
        )
        
        return JSONResponse(content={"predicted_digit": pred_digit})

    except ValueError as ve:
        # Client side image/value processing exceptions
        logger.warning(f"Data transformation error: {ve}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))

    except Exception as exc:
        # Backend systems exception safeguard
        logger.exception(f"Unexpected prediction operational failure: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The prediction runtime system hit an unexpected algorithmic execution anomaly."
        )