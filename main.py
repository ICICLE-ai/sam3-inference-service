import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sam3.ml_models import lifespan
from sam3.routes.predict import router

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=2128)
