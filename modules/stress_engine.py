"""
VisionGuard — Stress Engine Module
Applies Albumentations adversarial/environmental transforms to images.
"""

import sys
from pathlib import Path

import numpy as np

try:
    from PIL import Image
    import albumentations as A
    import cv2  # noqa: F401  (required by albumentations internally)
except ImportError as e:
    print(f"[StressEngine] Import error: {e}")
    sys.exit(1)

# Resolve output dir relative to THIS file so it is CWD-independent
_MODULE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = _MODULE_ROOT / "output" / "stressed_images"


class StressEngine:
    """Applies scenario-defined image degradation transforms."""

    # Map scenario transform names → Albumentations classes
    TRANSFORM_MAP = {
        "RandomRain": A.RandomRain,
        "RandomFog": A.RandomFog,
        "GaussNoise": A.GaussNoise,
        "MotionBlur": A.MotionBlur,
        "RandomBrightnessContrast": A.RandomBrightnessContrast,
        "RandomSnow": A.RandomSnow,
        "ImageCompression": A.ImageCompression,
        "CoarseDropout": A.CoarseDropout,
        "OpticalDistortion": A.OpticalDistortion,
    }

    def __init__(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def apply(self, image: Image.Image, scenario: dict) -> Image.Image:
        """
        Apply the scenario's transform to *image* and return a stressed PIL image.

        Side-effect: saves stressed image to output/stressed_images/{id}.jpg
        """
        try:
            img_np = self._pil_to_numpy(image)

            transform_name = scenario["transform"]
            params = scenario.get("params", {})
            scenario_id = scenario["id"]

            if transform_name == "Combined":
                transformed_np = self._apply_combined(img_np, params)
            else:
                transform = self._build_transform(transform_name, params)
                transformed_np = transform(image=img_np)["image"]

            stressed_pil = self._numpy_to_pil(transformed_np)

            # Persist stressed image (OUTPUT_DIR is absolute — CWD-safe)
            out_path = OUTPUT_DIR / f"{scenario_id}.jpg"
            stressed_pil.save(str(out_path), quality=95)

            return stressed_pil

        except Exception as e:
            raise RuntimeError(
                f"[StressEngine] transform '{scenario.get('transform')}' "
                f"(scenario {scenario.get('id')}) failed: {e}"
            ) from e

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _build_transform(self, name: str, params: dict) -> A.Compose:
        """Factory: map transform name → Albumentations Compose instance."""
        if name not in self.TRANSFORM_MAP:
            raise ValueError(
                f"[StressEngine] Unknown transform '{name}'. "
                f"Valid: {list(self.TRANSFORM_MAP.keys())}"
            )
        cls = self.TRANSFORM_MAP[name]
        sanitized = self._sanitize_params(params)
        try:
            transform = A.Compose([cls(**sanitized, p=1.0)])
            return transform
        except TypeError as e:
            raise TypeError(
                f"[StressEngine] Could not instantiate {name} with {sanitized}: {e}"
            ) from e

    def _apply_combined(self, img_np: np.ndarray, params: dict) -> np.ndarray:
        """S15 Combined Attack: Rain + GaussNoise + MotionBlur via A.Compose."""
        rain_params = self._sanitize_params(params.get("rain", {}))
        noise_params = self._sanitize_params(params.get("noise", {}))
        blur_params = self._sanitize_params(params.get("blur", {}))

        transform = A.Compose([
            A.RandomRain(**rain_params, p=1.0),
            A.GaussNoise(**noise_params, p=1.0),
            A.MotionBlur(**blur_params, p=1.0),
        ])
        return transform(image=img_np)["image"]

    @staticmethod
    def _sanitize_params(params: dict) -> dict:
        """
        Convert JSON-deserialized types to what Albumentations expects.

        - Lists → tuples  (required for range params like slant_range, fog_coef_range, etc.)
        - drop_color stays as a tuple of ints (Albumentations accepts both list and tuple)
        """
        out = {}
        for k, v in params.items():
            if isinstance(v, list):
                out[k] = tuple(v)
            else:
                out[k] = v
        return out

    @staticmethod
    def _pil_to_numpy(image: Image.Image) -> np.ndarray:
        """Convert PIL RGB image to uint8 numpy array (H, W, 3)."""
        return np.array(image.convert("RGB"), dtype=np.uint8)

    @staticmethod
    def _numpy_to_pil(arr: np.ndarray) -> Image.Image:
        """Convert numpy array back to PIL Image."""
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")
