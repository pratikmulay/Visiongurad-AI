"""
VisionGuard — Shared Pytest Fixtures (conftest.py)
All session-scoped fixtures to avoid redundant model loads/image loads.
"""

import json
import sys
from pathlib import Path

import pytest

# Allow imports from project root when running pytest from visionguard/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from PIL import Image
    from modules.model_loader import ModelLoader, ensure_sample_image
    from modules.stress_engine import StressEngine
except ImportError as e:
    pytest.exit(f"[conftest] Critical import failed: {e}", returncode=1)

SCENARIOS_PATH = Path(__file__).resolve().parents[1] / "config" / "scenarios.json"
ASSET_PATH = Path(__file__).resolve().parents[1] / "assets" / "sample_image.jpg"


# ──────────────────────────────────────────────────────────────────────
# Session-scoped fixtures
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_image() -> Image.Image:
    """Load (or download) the test image and return 224×224 PIL Image."""
    try:
        img_path = ensure_sample_image(str(ASSET_PATH))
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        return img
    except Exception as e:
        pytest.fail(f"[conftest] Could not load sample image: {e}")


@pytest.fixture(scope="session")
def model() -> ModelLoader:
    """Return a fully loaded ModelLoader instance."""
    try:
        return ModelLoader()
    except Exception as e:
        pytest.fail(f"[conftest] Could not load model: {e}")


@pytest.fixture(scope="session")
def stress_engine() -> StressEngine:
    """Return a StressEngine instance."""
    try:
        return StressEngine()
    except Exception as e:
        pytest.fail(f"[conftest] Could not create StressEngine: {e}")


@pytest.fixture(scope="session")
def scenarios() -> list:
    """Load all 15 scenarios from config/scenarios.json."""
    try:
        with open(SCENARIOS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 15, f"Expected 15 scenarios, got {len(data)}"
        return data
    except Exception as e:
        pytest.fail(f"[conftest] Could not load scenarios: {e}")


@pytest.fixture(scope="session")
def baseline_score(model, sample_image) -> float:
    """Return the baseline confidence score on the clean image."""
    try:
        result = model.get_baseline(sample_image)
        print(
            f"\n[conftest] Baseline: label='{result['label']}' "
            f"score={result['score']:.4f}"
        )
        return result["score"]
    except Exception as e:
        pytest.fail(f"[conftest] Could not compute baseline: {e}")
