from contextlib import asynccontextmanager

from fastapi import FastAPI
from transformers import Sam3Model, Sam3Processor, Sam3TrackerModel, Sam3TrackerProcessor

from sam3.config import DEVICE, MODEL_ID

sam3_model: Sam3Model | None = None
sam3_processor: Sam3Processor | None = None
sam3_tracker_model: Sam3TrackerModel | None = None
sam3_tracker_processor: Sam3TrackerProcessor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global sam3_model, sam3_processor, sam3_tracker_model, sam3_tracker_processor

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
