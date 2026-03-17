"""
VisionGuard — HTML Report Generator
Builds a self-contained safety scorecard using Jinja2.
"""

import json
import sys
import base64
from pathlib import Path
from datetime import datetime

try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
except ImportError as e:
    print(f"[ReportGenerator] Import error: {e}")
    sys.exit(1)

TEMPLATE_DIR = Path(__file__).resolve().parent
TEMPLATE_FILE = "template.html"


class ReportGenerator:
    """Renders the VisionGuard HTML safety scorecard."""

    def __init__(self):
        try:
            self.env = Environment(
                loader=FileSystemLoader(str(TEMPLATE_DIR)),
                autoescape=select_autoescape(["html"]),
            )
            self.template = self.env.get_template(TEMPLATE_FILE)
        except Exception as e:
            raise RuntimeError(
                f"[ReportGenerator] Failed to load Jinja2 template: {e}"
            ) from e

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(self, results: list, output_path: str = "output/report.html"):
        """
        Render the HTML report from results list and write to output_path.

        Args:
            results: list of result dicts from RobustnessAuditor.
            output_path: destination path for the HTML file.
        """
        try:
            stats = self._compute_stats(results)
            grouped = self._group_by_category(results)
            top_failures = self._get_top_failures(results, n=3)
            image_data = self._load_images_as_base64(top_failures)

            context = {
                "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model_name": "google/vit-base-patch16-224",
                "results": results,
                "stats": stats,
                "grouped": grouped,
                "top_failures": top_failures,
                "image_data": image_data,
                "failures": [r for r in results if r["status"] == "FAIL"],
            }

            html_content = self.template.render(**context)

            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            print(f"[ReportGenerator] Report saved → {out_path}")

        except Exception as e:
            raise RuntimeError(f"[ReportGenerator] Failed to generate report: {e}") from e

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _compute_stats(self, results: list) -> dict:
        """Compute summary statistics from results."""
        total = len(results)
        passed = sum(1 for r in results if r["status"] == "PASS")
        failed = total - passed
        pass_rate = (passed / total * 100) if total > 0 else 0.0
        drops = [r["confidence_drop"] for r in results]
        avg_drop = sum(drops) / len(drops) if drops else 0.0
        worst = max(results, key=lambda r: r["confidence_drop"], default=None)
        best = min(results, key=lambda r: r["confidence_drop"], default=None)
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": round(pass_rate, 1),
            "avg_drop": round(avg_drop, 4),
            "worst_scenario": worst,
            "best_scenario": best,
        }

    def _group_by_category(self, results: list) -> dict:
        """Group results by category."""
        grouped: dict[str, list] = {}
        for r in results:
            cat = r.get("category", "Other")
            grouped.setdefault(cat, []).append(r)
        return grouped

    def _get_top_failures(self, results: list, n: int = 3) -> list:
        """Return top N results with highest confidence drop."""
        sorted_results = sorted(
            results, key=lambda r: r["confidence_drop"], reverse=True
        )
        return sorted_results[:n]

    def _load_images_as_base64(self, scenarios: list) -> dict:
        """Load stressed images as base64 strings for embedding in HTML."""
        image_data = {}
        for r in scenarios:
            img_path = Path(r.get("stressed_image_path", ""))
            if img_path.exists():
                try:
                    with open(img_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")
                    image_data[r["scenario_id"]] = f"data:image/jpeg;base64,{b64}"
                except Exception:
                    image_data[r["scenario_id"]] = ""
            else:
                image_data[r["scenario_id"]] = ""
        return image_data
