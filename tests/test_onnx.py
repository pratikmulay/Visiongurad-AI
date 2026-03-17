"""
VisionGuard — ONNX Model Tests
Verifies the exported ONNX model exists, runs, and matches PyTorch output.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on sys.path so fixtures from conftest.py resolve
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ONNX_PATH = Path(__file__).resolve().parents[1] / "output" / "onnx" / "visionguard.onnx"


# ──────────────────────────────────────────────────────────────────────
# Test 1: File existence
# ──────────────────────────────────────────────────────────────────────

def test_onnx_file_exists():
    """The exported ONNX file must exist on disk."""
    assert ONNX_PATH.exists(), (
        f"ONNX file not found at '{ONNX_PATH}'. "
        "Run run_visionguard.py first to export the model."
    )
    size_mb = ONNX_PATH.stat().st_size / (1024 * 1024)
    assert size_mb > 10, (
        f"ONNX file seems too small ({size_mb:.1f} MB). "
        "Export may have been incomplete."
    )


# ──────────────────────────────────────────────────────────────────────
# Test 2: OnnxRuntime inference on dummy input
# ──────────────────────────────────────────────────────────────────────

def test_onnx_inference_runs():
    """
    Load the ONNX model with OnnxRuntime and run on a dummy [1, 3, 224, 224]
    input. Assert output shape is [1, 1000].
    """
    pytest.importorskip("onnxruntime", reason="onnxruntime not installed")
    import onnxruntime as ort

    if not ONNX_PATH.exists():
        pytest.skip("ONNX file not found — skipping inference test.")

    try:
        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3
        session = ort.InferenceSession(
            str(ONNX_PATH),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )

        dummy = np.zeros((1, 3, 224, 224), dtype=np.float32)
        outputs = session.run(None, {"pixel_values": dummy})

        out_shape = outputs[0].shape
        assert len(out_shape) == 2, (
            f"Expected 2-D output tensor, got shape {out_shape}"
        )
        assert out_shape[0] == 1, (
            f"Expected batch size 1, got {out_shape[0]}"
        )
        assert out_shape[1] == 1000, (
            f"Expected 1000 logits (ImageNet classes), got {out_shape[1]}"
        )

    except AssertionError:
        raise
    except Exception as e:
        pytest.fail(f"ONNX inference failed: {e}")


# ──────────────────────────────────────────────────────────────────────
# Test 3: ONNX output matches PyTorch output (top class index)
# ──────────────────────────────────────────────────────────────────────

def test_onnx_output_matches_pytorch(sample_image):
    """
    Run the same PIL image through both PyTorch pipeline and ONNX runtime.
    Assert the top predicted class index is identical.
    (Minor floating-point differences in scores are acceptable.)
    """
    pytest.importorskip("onnxruntime", reason="onnxruntime not installed")
    pytest.importorskip("torch", reason="torch not installed")
    pytest.importorskip("transformers", reason="transformers not installed")

    import torch
    import onnxruntime as ort
    # AutoImageProcessor replaces the deprecated AutoFeatureExtractor
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    if not ONNX_PATH.exists():
        pytest.skip("ONNX file not found — skipping PyTorch comparison test.")

    try:
        model_name = "google/vit-base-patch16-224"
        processor = AutoImageProcessor.from_pretrained(model_name)
        pt_model = AutoModelForImageClassification.from_pretrained(model_name)
        pt_model.eval()

        # Preprocess image with the official processor
        inputs = processor(images=sample_image, return_tensors="pt")
        pixel_values: torch.Tensor = inputs["pixel_values"]  # [1, 3, 224, 224]

        # PyTorch inference — no_grad prevents gradient tracking
        with torch.no_grad():
            pt_outputs = pt_model(pixel_values=pixel_values)
        pt_top_idx = int(pt_outputs.logits.argmax(dim=-1).item())

        # ONNX inference
        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3
        session = ort.InferenceSession(
            str(ONNX_PATH),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )

        # .detach().numpy() is required — plain .numpy() raises if grad is tracked
        onnx_outputs = session.run(
            None, {"pixel_values": pixel_values.detach().numpy()}
        )
        onnx_top_idx = int(np.argmax(onnx_outputs[0], axis=-1)[0])

        assert pt_top_idx == onnx_top_idx, (
            f"Top-class mismatch: PyTorch={pt_top_idx}, ONNX={onnx_top_idx}. "
            "Model export may have introduced numerical errors."
        )

    except AssertionError:
        raise
    except Exception as e:
        pytest.fail(f"PyTorch vs ONNX comparison failed: {e}")
