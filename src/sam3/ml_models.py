from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from sam3.config import DEVICE, MODEL_ID

# Typed as Any so this module is importable even before Sam3* classes
# land in a released transformers version. The lifespan function imports
# them lazily at startup so tests can mock these globals without issue.
sam3_model: Any = None
sam3_processor: Any = None
sam3_tracker_model: Any = None
sam3_tracker_processor: Any = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global sam3_model, sam3_processor, sam3_tracker_model, sam3_tracker_processor

    from transformers import Sam3Model, Sam3Processor, Sam3TrackerModel, Sam3TrackerProcessor

    print(f"Initializing SAM 3 Models on {DEVICE}...")

    try:
        print("Loading Sam3Model (Concept Segmentation)...")
        sam3_model = Sam3Model.from_pretrained(MODEL_ID).to(DEVICE)
        sam3_processor = Sam3Processor.from_pretrained(MODEL_ID)

        print("Loading Sam3TrackerModel (Visual Segmentation)...")
        sam3_tracker_model = Sam3TrackerModel.from_pretrained(MODEL_ID).to(DEVICE)
        sam3_tracker_processor = Sam3TrackerProcessor.from_pretrained(MODEL_ID)

        print("Both Models Loaded Successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    yield
    print("Shutting down...")
