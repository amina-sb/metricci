"""Simulation evidence for the README.

Draws data sets from a binormal model whose true AUC and F1 are known in closed
form, builds 95% intervals on each, and reports how often they cover the truth.
Also measures the size and power of the paired permutation test.

    python scripts/evidence.py            # print the tables
    python scripts/evidence.py --check    # exit 1 if README.md disagrees

Everything is seeded; the output is identical from run to run.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit
from scipy.stats import norm

from metricci import CIResult, auc_ci, f1_ci, hanley_mcneil_ci, paired_permutation_test, roc_auc

SEP = 1.0  # positives ~ N(SEP, 1), negatives ~ N(0, 1)
THRESHOLD = 0.5  # hard predictions for F1: score > THRESHOLD
SIZES = ((12, 28), (60, 140))  # (positives, negatives)
SEEDS = (11, 12, 13, 14, 15)
REPS_PER_SEED = 80
N_RESAMPLES = 999
README = Path(__file__).resolve().parent.parent / "README.md"
START, END = "<!-- evidence:start -->", "<!-- evidence:end -->"

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


def true_auc() -> float:
    return float(norm.cdf(SEP / math.sqrt(2)))


def true_f1(n_pos: int, n_neg: int) -> float:
    pi = n_pos / (n_pos + n_neg)
    tpr = 1 - norm.cdf(THRESHOLD - SEP)
    fpr = 1 - norm.cdf(THRESHOLD)
    tp, fp, fn = pi * tpr, (1 - pi) * fpr, pi * (1 - tpr)
    return float(2 * tp / (2 * tp + fp + fn))


def draw(rng: np.random.Generator, n_pos: int, n_neg: int) -> tuple[IntArray, FloatArray]:
    y = np.r_[np.ones(n_pos, dtype=np.int64), np.zeros(n_neg, dtype=np.int64)]
    s = np.r_[rng.normal(SEP, 1, n_pos), rng.normal(0, 1, n_neg)]
    return y, s


def wilson(hits: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


@dataclass
class Tally:
    hits: list[int]
    widths: list[float]

    def row(self, label: str, method: str) -> str:
        total = sum(self.hits)
        n = len(SEEDS) * REPS_PER_SEED
        lo, hi = wilson(total, n)
        per_seed = [h / REPS_PER_SEED for h in self.hits]
        return (
            f"| {label} | {method} | {total / n:.3f} ({lo:.3f}-{hi:.3f}) "
            f"| {min(per_seed):.3f}-{max(per_seed):.3f} | {np.mean(self.widths):.3f} |"
        )


Builder = Callable[[IntArray, FloatArray, np.random.Generator], CIResult]


def coverage(builders: dict[str, Builder], n_pos: int, n_neg: int, truth: float) -> list[str]:
    tallies = {m: Tally([], []) for m in builders}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        counts = dict.fromkeys(builders, 0)
        for _ in range(REPS_PER_SEED):
            y, s = draw(rng, n_pos, n_neg)
            rep_seed = int(rng.integers(2**31))
            for m, build in builders.items():
                res = build(y, s, np.random.default_rng(rep_seed))
                counts[m] += res.low <= truth <= res.high
                tallies[m].widths.append(res.high - res.low)
        for m in builders:
            tallies[m].hits.append(counts[m])
    label = f"n = {n_pos + n_neg} ({n_pos} pos)"
    return [t.row(label, m) for m, t in tallies.items()]


def auc_table() -> list[str]:
    builders: dict[str, Builder] = {
        "Hanley-McNeil (baseline)": lambda y, s, r: hanley_mcneil_ci(y, s),
        "percentile": lambda y, s, r: auc_ci(
            y, s, method="percentile", n_resamples=N_RESAMPLES, random_state=r
        ),
        "basic": lambda y, s, r: auc_ci(
            y, s, method="basic", n_resamples=N_RESAMPLES, random_state=r
        ),
        "BCa": lambda y, s, r: auc_ci(y, s, method="bca", n_resamples=N_RESAMPLES, random_state=r),
    }
    rows = []
    for n_pos, n_neg in SIZES:
        rows += coverage(builders, n_pos, n_neg, true_auc())
    return rows


def f1_table() -> list[str]:
    def make(method: str) -> Builder:
        return lambda y, s, r: f1_ci(
            y, (s > THRESHOLD).astype(np.int64), method=method,  # type: ignore[arg-type]
            n_resamples=N_RESAMPLES, random_state=r,
        )  # fmt: skip

    builders = {m: make(m.lower()) for m in ("percentile", "basic", "BCa")}
    rows = []
    for n_pos, n_neg in SIZES:
        rows += coverage(builders, n_pos, n_neg, true_f1(n_pos, n_neg))
    return rows


def permutation_table(reps: int = 200, n_perm: int = 499) -> list[str]:
    """Rejection rate at alpha = 0.05 when model B outputs margins on a wide
    scale and model A outputs probabilities."""
    rows = []
    for gain, label in ((0.0, "equal models (size)"), (0.5, "A better by 0.5 SD (power)")):
        for rank in (False, True):
            rng = np.random.default_rng(99)
            rejected = 0
            for _ in range(reps):
                y = np.r_[np.ones(60, dtype=np.int64), np.zeros(140, dtype=np.int64)]
                shared = rng.normal(0, 1, 200)
                a = expit(y * (1.0 + gain) + shared + rng.normal(0, 0.5, 200))
                b = 25.0 * (y * 1.0 + shared + rng.normal(0, 0.5, 200))
                res = paired_permutation_test(
                    roc_auc, y, a, b, n_permutations=n_perm, rank_transform=rank,
                    random_state=rng,
                )  # fmt: skip
                rejected += res.pvalue <= 0.05
            lo, hi = wilson(rejected, reps)
            how = "ranks (default)" if rank else "raw scores"
            rows.append(f"| {label} | {how} | {rejected / reps:.3f} ({lo:.3f}-{hi:.3f}) |")
    return rows


def render() -> str:
    n = len(SEEDS) * REPS_PER_SEED
    head = "| data | method | coverage (95% CI) | range over 5 seeds | mean width |"
    rule = "|---|---|---|---|---|"
    parts = [
        f"**AUC**, true value {true_auc():.4f}. Nominal 95% intervals, {n} simulated "
        f"data sets per row, {N_RESAMPLES} resamples each.",
        "",
        head,
        rule,
        *auc_table(),
        "",
        f"**F1** at threshold {THRESHOLD}, true value {true_f1(*SIZES[0]):.4f} "
        "(both sizes have a positive rate of 0.3).",
        "",
        head,
        rule,
        *f1_table(),
        "",
        "**Paired permutation test** for an AUC difference, 200 data sets per row, "
        "n = 200, 499 permutations, rejection rate at alpha = 0.05.",
        "",
        "| scenario | swapped values | rejection rate (95% CI) |",
        "|---|---|---|",
        *permutation_table(),
    ]
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare with README.md")
    args = parser.parse_args()
    table = render()
    if not args.check:
        print(table)
        return 0
    text = README.read_text()
    try:
        published = text.split(START, 1)[1].split(END, 1)[0].strip()
    except IndexError:
        print("README.md has no evidence markers", file=sys.stderr)
        return 1
    if published != table.strip():
        print("README evidence table is out of date. Regenerated table:\n", file=sys.stderr)
        print(table, file=sys.stderr)
        return 1
    print("README evidence table matches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
