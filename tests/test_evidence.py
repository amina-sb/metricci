from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import evidence  # noqa: E402


def test_closed_form_truths_match_a_large_sample() -> None:
    rng = np.random.default_rng(0)
    y, s = evidence.draw(rng, 60_000, 140_000)
    from metricci import f1, roc_auc

    assert float(roc_auc(y, s)) == pytest.approx(evidence.true_auc(), abs=0.003)
    pred = (s > evidence.THRESHOLD).astype(np.int64)
    assert float(f1(y, pred)) == pytest.approx(evidence.true_f1(60, 140), abs=0.003)


def test_true_lift_matches_large_samples() -> None:
    from metricci import lift_at_k

    values = [
        float(lift_at_k(*evidence.draw(np.random.default_rng(k), 60_000, 140_000), 0.1))
        for k in range(4)
    ]
    assert float(np.mean(values)) == pytest.approx(evidence.true_lift(60, 140), abs=0.015)


def test_wilson_interval_contains_proportion() -> None:
    lo, hi = evidence.wilson(380, 400)
    assert lo < 0.95 < hi


@pytest.mark.evidence
def test_readme_table_is_reproduced() -> None:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "evidence.py"), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_readme_example_output_is_current(capsys: pytest.CaptureFixture[str]) -> None:
    """Run the first python block in the README and compare what it prints with
    the ``# ...`` comment under each print call."""
    text = (ROOT / "README.md").read_text()
    code = text.split("```python\n", 1)[1].split("```", 1)[0]
    expected = [ln[2:] for ln in code.splitlines() if ln.startswith("# ")]
    exec(compile(code, "README.md", "exec"), {})
    assert capsys.readouterr().out.splitlines() == expected
