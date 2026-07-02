from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List


class SegmentationRequest(BaseModel):
    image_id: str
    pipe_id: str
    system_id: str
    image_path: str
    text_prompts: list[str] | None = None
    x: int | None = None
    y: int | None = None
    patch_size: int | None = None
    crop_size: int | None = None
    overlap_ratio: float = 0.2
    threshold: float = 0.1
    mask_threshold: float = 0.1

    @field_validator("text_prompts")
    @classmethod
    def validate_text_prompts(cls, v):
        if v is not None:
            v = [p.strip() for p in v if p and p.strip()]
            if len(v) == 0:
                return None
        return v

    @field_validator("patch_size", "crop_size")
    @classmethod
    def validate_patch_sizes(cls, v):
        if v is not None and v <= 0:
            raise ValueError("patch_size/crop_size must be > 0")
        return v

    @field_validator("overlap_ratio")
    @classmethod
    def validate_overlap_ratio(cls, v):
        if v < 0 or v >= 1:
            raise ValueError("overlap_ratio must be in [0, 1)")
        return v

    @field_validator("threshold", "mask_threshold")
    @classmethod
    def validate_thresholds(cls, v):
        if v < 0 or v > 1:
            raise ValueError("threshold/mask_threshold must be in [0, 1]")
        return v

    @model_validator(mode="after")
    def validate_patch_aliases(self):
        if (
            self.patch_size is not None
            and self.crop_size is not None
            and self.patch_size != self.crop_size
        ):
            raise ValueError(
                "If both patch_size and crop_size are provided, they must be equal"
            )
        return self

    def get_effective_patch_size(self) -> int | None:
        return self.patch_size if self.patch_size is not None else self.crop_size

    class Config:
        json_schema_extra = {
            "example": {
                "image_id": "img_123",
                "pipe_id": "pipe_456",
                "system_id": "tapis_system",
                "image_path": "/path/to/image.jpg",
                "text_prompts": ["red bottle", "person", "chair"],
                "patch_size": 960,
                "overlap_ratio": 0.2,
            }
        }


class BoundingBox(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    confidence: float
    label: str
    prompt_index: int
    segmentation: List[List[int]] = Field(default_factory=list)


class SegmentationResponse(BaseModel):
    bboxes: List[BoundingBox]
    prompt_type: str
    prompt_count: int
    total_detections: int
    model: str
    total_time_seconds: float
    per_prompt_breakdown: Optional[dict] = None


class ExemplarRequest(BaseModel):
    image_id: str
    pipe_id: str
    system_id: str
    image_path: str
    exemplar_box: List[int] = Field(..., min_length=4, max_length=4)
    patch_size: int | None = None
    overlap_ratio: float = 0.2
    threshold: float = 0.1
    mask_threshold: float = 0.1

    @field_validator("exemplar_box")
    @classmethod
    def validate_exemplar_box(cls, v):
        if len(v) != 4:
            raise ValueError("exemplar_box must have exactly 4 integers: [x1, y1, x2, y2]")
        x1, y1, x2, y2 = v
        if x1 >= x2 or y1 >= y2:
            raise ValueError("exemplar_box must satisfy x1 < x2 and y1 < y2")
        if any(c < 0 for c in v):
            raise ValueError("exemplar_box coordinates must be non-negative")
        return v

    @field_validator("patch_size")
    @classmethod
    def validate_patch_size(cls, v):
        if v is not None and v <= 0:
            raise ValueError("patch_size must be > 0")
        return v

    @field_validator("overlap_ratio")
    @classmethod
    def validate_overlap_ratio(cls, v):
        if v < 0 or v >= 1:
            raise ValueError("overlap_ratio must be in [0, 1)")
        return v

    @field_validator("threshold", "mask_threshold")
    @classmethod
    def validate_thresholds(cls, v):
        if v < 0 or v > 1:
            raise ValueError("threshold/mask_threshold must be in [0, 1]")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "image_id": "img_123",
                "pipe_id": "pipe_456",
                "system_id": "tapis_system",
                "image_path": "/path/to/image.jpg",
                "exemplar_box": [120, 80, 240, 200],
                "patch_size": 960,
                "overlap_ratio": 0.2,
                "threshold": 0.1,
            }
        }


class LegacyPointRequest(BaseModel):
    """Legacy single-point request for backwards compatibility."""
    image_id: str
    pipe_id: str
    system_id: str
    image_path: str
    x: int
    y: int
