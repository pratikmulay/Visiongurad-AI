"""
VisionGuard — Model Loader Module
Loads google/vit-base-patch16-224 via HuggingFace Transformers pipeline.
"""

import os
import sys
from pathlib import Path

try:
    from PIL import Image
    import requests
    from transformers import pipeline
except ImportError as e:
    print(f"[ModelLoader] Import error: {e}")
    sys.exit(1)

# Default fallback image URL if local sample not found
FALLBACK_IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/"
    "1/14/Gatto_europeo4.jpg/320px-Gatto_europeo4.jpg"
)
MODEL_NAME = "google/vit-base-patch16-224"


class ModelLoader:
    """Loads and wraps the ViT classification pipeline."""

    _instance = None  # singleton cache

    def __init__(self):
        self.model_name = MODEL_NAME
        self._pipeline = None
        self.baseline = None
        self._load_model()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_model(self):
        """Load the HuggingFace pipeline (cached after first call)."""
        try:
            print(f"[ModelLoader] Loading '{self.model_name}' …")
            self._pipeline = pipeline(
                "image-classification",
                model=self.model_name,
            )
            print("VisionGuard: Model loaded successfully.")
        except Exception as e:
            raise RuntimeError(
                f"[ModelLoader] Failed to load model '{self.model_name}': {e}"
            ) from e

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, image: Image.Image) -> dict:
        """
        Run inference on a PIL image.

        Returns:
            {"label": str, "score": float}  — top-1 prediction only.
        """
        try:
            img = image.convert("RGB").resize((224, 224))
            results = self._pipeline(img, top_k=1)
            top = results[0]
            return {"label": top["label"], "score": float(top["score"])}
        except Exception as e:
            raise RuntimeError(f"[ModelLoader] Prediction failed: {e}") from e

    def get_baseline(self, image: Image.Image) -> dict:
        """
        Predict on clean image and cache result as self.baseline.

        Returns:
            {"label": str, "score": float}
        """
        try:
            self.baseline = self.predict(image)
            return self.baseline
        except Exception as e:
            raise RuntimeError(
                f"[ModelLoader] Baseline prediction failed: {e}"
            ) from e


# ------------------------------------------------------------------
# Utility: ensure sample image exists (downloads or generates fallback)
# ------------------------------------------------------------------
def ensure_sample_image(asset_path: str = "assets/sample_image.jpg") -> str:
    """
    Return path to a valid sample image.
    Priority: (1) existing file, (2) network download, (3) synthetic generation.

    Args:
        asset_path: relative or absolute path where image should exist.

    Returns:
        Resolved absolute path string.
    """
    path = Path(asset_path)
    if path.exists():
        return str(path.resolve())

    print(f"[ModelLoader] '{asset_path}' not found. Attempting download …")
    path.parent.mkdir(parents=True, exist_ok=True)

    # Try multiple fallback URLs
    fallback_urls = [
        FALLBACK_IMAGE_URL,
        "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png",
    ]
    for url in fallback_urls:
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            with open(path, "wb") as f:
                f.write(resp.content)
            print(f"[ModelLoader] Downloaded fallback from {url[:60]}")
            return str(path.resolve())
        except Exception:
            continue

    # Last resort: generate a synthetic image locally (no network needed)
    print("[ModelLoader] Network unavailable — generating synthetic test image.")
    try:
        _generate_synthetic_image(path)
        print(f"[ModelLoader] Synthetic image saved to '{path}'.")
        return str(path.resolve())
    except Exception as e:
        raise RuntimeError(
            f"[ModelLoader] Could not create any test image: {e}"
        ) from e


def _generate_synthetic_image(path: Path) -> None:
    """Generate a simple 320×240 RGB JPEG for testing purposes."""
    img = Image.new("RGB", (320, 240), color=(135, 170, 220))
    try:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        # Simple cat-like shape so ViT returns non-trivial results
        draw.rectangle([0, 160, 320, 240], fill=(80, 120, 60))
        draw.ellipse([100, 100, 220, 180], fill=(200, 180, 140))
        draw.ellipse([130, 60, 200, 130], fill=(200, 180, 140))
        draw.polygon([(135, 80), (145, 50), (160, 78)], fill=(200, 180, 140))
        draw.polygon([(175, 78), (190, 50), (200, 80)], fill=(200, 180, 140))
        draw.ellipse([145, 85, 160, 100], fill=(50, 150, 50))
        draw.ellipse([170, 85, 185, 100], fill=(50, 150, 50))
    except Exception:
        pass  # fall back to plain coloured rectangle
    img.save(str(path), quality=95)
