from typing import Literal
from pydantic import BaseModel, Field, field_validator

class MetadataInput(BaseModel):
    pen_pressure: float = Field(..., ge=0.0, le=10.0, description="Pressure value between 0 and 10.")
    writer_age: int = Field(..., ge=1, le=120, description="A valid human age between 1 and 120.")
    handedness: Literal["left", "right", "ambidextrous"]

    @classmethod
    def from_form(cls, pen_pressure: float, writer_age: int, handedness: str):
        """Helper to initialize schema validation from FastAPI Form fields"""
        return cls(pen_pressure=pen_pressure, writer_age=writer_age, handedness=handedness)