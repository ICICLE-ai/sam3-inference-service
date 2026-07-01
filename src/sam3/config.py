import os
import torch

TAPIS_BASE_URL = "https://icicleai.tapis.io"
MODEL_ID = "facebook/sam3"

is_cuda = torch.cuda.is_available()
print(f"CUDA Available: {is_cuda}")
DEVICE = "cuda" if is_cuda else "cpu"

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_TTL = 3600  # seconds
REDIS_EMBEDDING_KEY_PREFIX = "sam3:emb"
