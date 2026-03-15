"""
VisionGuard — Robustness Auditor Module
Measures confidence drop across all 15 adversarial scenarios.
"""

import json
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError as e:
    print(f"[Auditor] Import error: {e}")
    sys.exit(1)

OUTPUT_DIR = Path("output")


class RobustnessAuditor:
    """Runs all test scenarios and records robustness metrics."""

    def __init__(self, model_loader, stress_engine):
        self.model = model_loader
        self.engine = stress_engine
        self.results: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_scenario(self, image: Image.Image, scenario: dict) -> dict:
        """
        Run one scenario and return a result dict.

        Steps:
        1. Baseline prediction on clean image.
        2. Apply stress transform.
        3. Prediction on stressed image.
        4. Compute confidence drop.
        5. Determine PASS / FAIL.
        """
        try:
            # 1. Baseline
            baseline = self.model.get_baseline(image)

            # 2. Apply stress
            stressed_img = self.engine.apply(image, scenario)

            # 3. Stressed prediction
            stressed = self.model.predict(stressed_img)

            # 4. Confidence drop (floor at 0 to avoid negative drops)
            drop = max(0.0, baseline["score"] - stressed["score"])

            # 5. Status
            threshold = scenario["pass_threshold"]
            status = "PASS" if drop < threshold else "FAIL"

            # Stressed image path
            stressed_path = str(
                Path("output/stressed_images") / f"{scenario['id']}.jpg"
            )

            result = {
                "scenario_id": scenario["id"],
                "scenario_name": scenario["name"],
                "category": scenario["category"],
                "severity": scenario["severity"],
                "baseline_label": baseline["label"],
                "baseline_score": round(baseline["score"], 4),
                "stressed_label": stressed["label"],
                "stressed_score": round(stressed["score"], 4),
                "confidence_drop": round(drop, 4),
                "pass_threshold": threshold,
                "status": status,
                "stressed_image_path": stressed_path,
            }

            self.results.append(result)
            return result

        except Exception as e:
            # Record error scenario so the run continues
            error_result = {
                "scenario_id": scenario.get("id", "???"),
                "scenario_name": scenario.get("name", "Unknown"),
                "category": scenario.get("category", "Unknown"),
                "severity": scenario.get("severity", "unknown"),
                "baseline_label": "ERROR",
                "baseline_score": 0.0,
                "stressed_label": "ERROR",
                "stressed_score": 0.0,
                "confidence_drop": 0.0,
                "pass_threshold": scenario.get("pass_threshold", 1.0),
                "status": "ERROR",
                "stressed_image_path": "",
                "error": str(e),
            }
            self.results.append(error_result)
            print(
                f"[Auditor] ⚠  Scenario {scenario.get('id')} failed with error: {e}"
            )
            return error_result

    def run_all(self, image: Image.Image, scenarios: list) -> list:
        """
        Run all scenarios sequentially, printing progress.

        Returns:
            List of result dicts.
        """
        self.results = []
        total = len(scenarios)
        for i, scenario in enumerate(scenarios, 1):
            print(
                f"  Testing {scenario['id']}/{total:02d}: {scenario['name']} …",
                flush=True,
            )
            self.run_scenario(image, scenario)
            result = self.results[-1]
            status_icon = "✓" if result["status"] == "PASS" else "✗"
            print(
                f"    {status_icon} status={result['status']} | "
                f"drop={result['confidence_drop']:.4f} | "
                f"threshold={result['pass_threshold']}"
            )
        return self.results

    def save_results(self, path: str = "output/results.json"):
        """Persist self.results to JSON."""
        try:
            out_path = Path(path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2)
            print(f"[Auditor] Results saved → {out_path}")
        except Exception as e:
            raise RuntimeError(f"[Auditor] Failed to save results: {e}") from e
