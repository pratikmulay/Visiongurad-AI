"""
VisionGuard — Robustness Test Suite
Parametrized across all 15 adversarial/environmental scenarios.
"""

import json
from pathlib import Path

import pytest

# ──────────────────────────────────────────────────────────────────────
# Load scenarios at module level for parametrize
# ──────────────────────────────────────────────────────────────────────

_SCENARIOS_PATH = Path(__file__).resolve().parents[1] / "config" / "scenarios.json"

try:
    with open(_SCENARIOS_PATH, "r", encoding="utf-8") as _f:
        _scenarios_list: list = json.load(_f)
except Exception as _e:
    _scenarios_list = []
    print(f"[test_robustness] WARNING: Could not load scenarios: {_e}")


def _get_scenario_ids() -> list[str]:
    """Return scenario IDs for use as pytest parametrize labels."""
    return [s.get("id", f"S??") for s in _scenarios_list]


# ──────────────────────────────────────────────────────────────────────
# Parametrized robustness tests (one per scenario)
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("scenario", _scenarios_list, ids=_get_scenario_ids())
def test_robustness_scenario(
    scenario, sample_image, model, stress_engine, baseline_score
):
    """
    Tests that model confidence drop under a given adversarial scenario
    stays within the defined safety threshold.

    PASS: confidence_drop < pass_threshold  →  model is robust
    FAIL: confidence_drop ≥ pass_threshold  →  robustness failure detected
    """
    try:
        # Apply transform
        stressed = stress_engine.apply(sample_image, scenario)

        # Inference on stressed image
        result = model.predict(stressed)

        # Compute drop (floored at 0)
        drop = max(0.0, baseline_score - result["score"])
        threshold = scenario["pass_threshold"]

        assert drop < threshold, (
            f"ROBUSTNESS FAILURE — {scenario['name']} [{scenario['id']}]: "
            f"Confidence dropped {drop:.1%} "
            f"(max allowed: {threshold:.1%}). "
            f"Baseline: {baseline_score:.1%} → "
            f"Stressed: {result['score']:.1%}. "
            f"Category: {scenario['category']} | "
            f"Severity: {scenario['severity']}"
        )

    except AssertionError:
        raise
    except Exception as e:
        pytest.fail(
            f"[test_robustness] Unexpected error in scenario "
            f"{scenario.get('id','?')} '{scenario.get('name','?')}': {e}"
        )


# ──────────────────────────────────────────────────────────────────────
# Additional standalone robustness tests
# ──────────────────────────────────────────────────────────────────────

def test_baseline_confidence_acceptable(model, sample_image):
    """
    Baseline prediction on the clean image must exceed 70% confidence.
    A low baseline indicates a poor model–image fit before any stress.
    """
    result = model.get_baseline(sample_image)
    score = result["score"]
    assert score > 0.70, (
        f"Baseline confidence too low: {score:.1%}. "
        "Model may not recognise the test image correctly."
    )


def test_model_label_consistency_under_low_stress(
    model, stress_engine, sample_image, scenarios
):
    """
    Under LOW-severity scenarios the predicted label must not change,
    even if the confidence score drops.
    """
    baseline = model.get_baseline(sample_image)
    baseline_label = baseline["label"]

    low_severity = [s for s in scenarios if s.get("severity") == "low"]
    assert low_severity, "No low-severity scenarios found — check scenarios.json."

    failures = []
    for scenario in low_severity:
        try:
            stressed = stress_engine.apply(sample_image, scenario)
            result = model.predict(stressed)
            if result["label"] != baseline_label:
                failures.append(
                    f"{scenario['id']} ({scenario['name']}): "
                    f"'{baseline_label}' → '{result['label']}'"
                )
        except Exception as e:
            failures.append(f"{scenario['id']}: error — {e}")

    assert not failures, (
        f"Label changed under LOW-severity stress in {len(failures)} scenario(s):\n"
        + "\n".join(f"  • {f}" for f in failures)
    )


def test_all_15_scenarios_execute_without_error(stress_engine, sample_image, scenarios):
    """
    Verifies that all 15 transforms execute without raising exceptions.
    Does NOT check confidence — only that transforms complete successfully.
    """
    assert len(scenarios) == 15, (
        f"Expected 15 scenarios in scenarios.json, found {len(scenarios)}."
    )
    errors = []
    for scenario in scenarios:
        try:
            stress_engine.apply(sample_image, scenario)
        except Exception as e:
            errors.append(f"{scenario['id']} ({scenario['name']}): {e}")

    assert not errors, (
        f"{len(errors)} scenario(s) raised exceptions:\n"
        + "\n".join(f"  • {err}" for err in errors)
    )
