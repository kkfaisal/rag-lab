"""Is run B really better than run A, or is the difference noise?

Two runs on the same questions give paired scores: (a_i, b_i) per question. We report
  delta      mean(b - a)
  ci95       a 95% bootstrap confidence interval for that mean (resampling questions)
  p_value    a paired permutation test: if A and B were equally good, swapping a_i and
             b_i on any question would be as likely as not; p is how often random swaps
             give a mean difference at least as large as the one observed
Both use a fixed seed, so the same two runs always give the same numbers.
"""

import numpy as np

RESAMPLES = 10_000
MIN_QUESTIONS = 30        # below this, even large differences are often noise


def paired(a: list[float], b: list[float], alpha: float = 0.05) -> dict:
    x = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    n = len(x)
    out = {"n": n, "mean_a": None, "mean_b": None, "delta": None, "ci95": None, "p_value": None,
           "b_better": int((x > 0).sum()), "a_better": int((x < 0).sum()), "ties": int((x == 0).sum()),
           "significant": False, "verdict": "no paired questions"}
    if n == 0:
        return out
    rng = np.random.default_rng(0)
    delta = float(x.mean())
    boot = x[rng.integers(0, n, size=(RESAMPLES, n))].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    if np.all(x == 0):
        p = 1.0
    else:
        signs = rng.choice([-1.0, 1.0], size=(RESAMPLES, n))
        p = float(((np.abs((signs * x).mean(axis=1)) >= abs(delta) - 1e-12).sum() + 1) / (RESAMPLES + 1))
    significant = p < alpha
    if significant:
        verdict = f"B is {'better' if delta > 0 else 'worse'} than A, beyond noise (p = {p:.3g})"
    else:
        verdict = f"No significant difference (p = {p:.2g}): this gap could be noise"
    if n < MIN_QUESTIONS:
        verdict += f". Only {n} questions: use more for a reliable comparison"
    out.update(mean_a=round(float(np.mean(a)), 2), mean_b=round(float(np.mean(b)), 2), delta=round(delta, 2),
               ci95=[round(float(lo), 2), round(float(hi), 2)], p_value=round(p, 4), significant=significant, verdict=verdict)
    return out
