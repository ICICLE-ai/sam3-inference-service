import io

import requests
from fastapi import HTTPException
from PIL import Image

from sam3.config import TAPIS_BASE_URL


def fetch_image_from_tapis(system_id: str, path: str, token: str) -> Image.Image:
    """Downloads an image from Tapis directly into memory."""
    clean_path = path.lstrip("/")
    url = f"{TAPIS_BASE_URL}/v3/files/content/{system_id}/{clean_path}"
    headers = {"X-Tapis-Token": token}

    print(f"Downloading: {clean_path}")
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Tapis Error: {resp.text}",
        )

    return Image.open(io.BytesIO(resp.content)).convert("RGB")
