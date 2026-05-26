import pytest
from pydantic import ValidationError
from app.schemas import MetadataInput

def test_metadata_input_valid():
    """Ensures that valid data successfully creates the schema object."""
    data = MetadataInput.from_form(pen_pressure=4.5, writer_age=25, handedness="right")
    assert data.pen_pressure == 4.5
    assert data.writer_age == 25
    assert data.handedness == "right"

def test_metadata_input_invalid_age():
    """Ensures that invalid writer_age raises a ValidationError."""
    with pytest.raises(ValidationError):
        MetadataInput.from_form(pen_pressure=4.5, writer_age=150, handedness="right")

def test_metadata_input_invalid_handedness():
    """Ensures that incorrect category throws a ValidationError."""
    with pytest.raises(ValidationError):
        MetadataInput.from_form(pen_pressure=4.5, writer_age=25, handedness="incorrect_string")

def test_metadata_input_invalid_pen_pressure():
    """Ensures that incorrect pen_pressure raises a ValidationError."""
    with pytest.raises(ValidationError):
        MetadataInput.from_form(pen_pressure=100.0, writer_age=25, handedness="left")