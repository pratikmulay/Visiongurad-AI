"""
VisionGuard — ONNX Exporter Module
Exports google/vit-base-patch16-224 to ONNX format and verifies it via OnnxRuntime.
"""

import sys
from pathlib import Path

import numpy as np

try:
    import torch
    from transformers import AutoModelForImageClassification, AutoImageProcessor
    import onnx
    import onnxruntime as ort
except ImportError as e:
    print(f"[ONNXExporter] Import error: {e}")
    sys.exit(1)

MODEL_NAME = "google/vit-base-patch16-224"

# Resolve output path relative to this file so it is CWD-independent
_MODULE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = _MODULE_ROOT / "output" / "onnx" / "visionguard.onnx"


class ONNXExporter:
    """Exports the ViT model to ONNX and validates the export."""

    def __init__(self, output_path: str | None = None):
        self.output_path = Path(output_path) if output_path else OUTPUT_PATH
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def export(self, hf_model_name: str = MODEL_NAME) -> str:
        """
        Export the HuggingFace ViT model to ONNX.

        Returns:
            str: Path to the exported .onnx file.
        """
        try:
            print(f"[ONNXExporter] Loading model '{hf_model_name}' for export …")

            # AutoImageProcessor replaces deprecated AutoFeatureExtractor
            _ = AutoImageProcessor.from_pretrained(hf_model_name)
            model = AutoModelForImageClassification.from_pretrained(hf_model_name)
            model.eval()

            # Dummy input — shape [batch=1, channels=3, height=224, width=224]
            dummy_input = torch.zeros(1, 3, 224, 224)

            print(f"[ONNXExporter] Exporting to '{self.output_path}' (opset 14) …")

            # torch.onnx.export requires a positional tensor arg, not a dict-in-tuple.
            # For HuggingFace models that accept keyword pixel_values, we wrap the
            # model in a tiny shim so the export call stays clean.
            class _PixelValuesShim(torch.nn.Module):
                def __init__(self, inner):
                    super().__init__()
                    self.inner = inner

                def forward(self, pixel_values):
                    return self.inner(pixel_values=pixel_values).logits

            shim = _PixelValuesShim(model)
            shim.eval()

            with torch.no_grad():
                torch.onnx.export(
                    shim,
                    dummy_input,           # plain tensor — no dict wrapper
                    str(self.output_path),
                    opset_version=14,
                    input_names=["pixel_values"],
                    output_names=["logits"],
                    dynamic_axes={
                        "pixel_values": {0: "batch_size"},
                        "logits": {0: "batch_size"},
                    },
                    export_params=True,
                    do_constant_folding=True,
                )

            self._print_size()
            print("[ONNXExporter] ONNX export complete. TensorRT-ready.")
            return str(self.output_path)

        except Exception as e:
            raise RuntimeError(f"[ONNXExporter] Export failed: {e}") from e

    def verify(self) -> bool:
        """
        Run OnnxRuntime on a dummy input and check output shape.

        Returns:
            True if output shape is (1, 1000), False otherwise.
        """
        try:
            if not self.output_path.exists():
                print(f"[ONNXExporter] ONNX file not found: {self.output_path}")
                return False

            # Validate ONNX graph structure
            onnx_model = onnx.load(str(self.output_path))
            onnx.checker.check_model(onnx_model)

            # OnnxRuntime inference (uses InferenceSession — not onnx.load)
            sess_opts = ort.SessionOptions()
            sess_opts.log_severity_level = 3  # suppress verbose logs
            session = ort.InferenceSession(
                str(self.output_path),
                sess_options=sess_opts,
                providers=["CPUExecutionProvider"],
            )

            dummy = np.zeros((1, 3, 224, 224), dtype=np.float32)
            outputs = session.run(None, {"pixel_values": dummy})
            out_shape = outputs[0].shape  # expected (1, 1000)

            if len(out_shape) == 2 and out_shape[1] == 1000:
                print(
                    f"[ONNXExporter] Verification ✓ — output shape: {out_shape}"
                )
                return True
            else:
                print(
                    f"[ONNXExporter] Verification ✗ — unexpected shape: {out_shape}"
                )
                return False

        except Exception as e:
            print(f"[ONNXExporter] Verification error: {e}")
            return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _print_size(self):
        """Print exported file size in MB."""
        try:
            size_mb = self.output_path.stat().st_size / (1024 * 1024)
            print(f"[ONNXExporter] File size: {size_mb:.1f} MB")
        except Exception:
            pass
