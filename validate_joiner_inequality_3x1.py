# ============================================================
# validate_sod_inequality_3x1.py
# (copy/paste exactly)
#
# Purpose:
#   Produce a 3-by-1 validation figure for SOD inequality
#
# Notes:
#   - Uses exact divisor-sum sieve
#   - Uses SPF sieve for prime supports
#   - No smoothing, no approximations
# ============================================================

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigma_sieve(N: int) -> np.ndarray:
    """Exact computation of σ(n) via divisor-sum sieve."""
    sig = np.zeros(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sig[d::d] += d
    return sig


def spf_sieve(N: int) -> np.ndarray:
    """Smallest-prime-factor sieve."""
    spf = np.zeros(N + 1, dtype=np.int32)
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
            if i * i <= N:
                for j in range(i * i, N + 1, i):
                    if spf[j] == 0:
                        spf[j] = i
    return spf


def distinct_primes_from_spf(n: int, spf: np.ndarray) -> List[int]:
    """Return distinct prime divisors of n using SPF."""
    primes: List[int] = []
    last = 0
    while n > 1:
        p = int(spf[n]) if spf[n] != 0 else n
        if p != last:
            primes.append(p)
            last = p
        while n % p == 0:
            n //= p
    return primes


def LU_from_prime_support(primes: List[int]) -> Tuple[float, float, int]:
    """Compute lower/upper prime-support envelopes."""
    if not primes:
        return 1.0, 1.0, 0
    arr = np.array(primes, dtype=np.float64)
    L = float(np.prod((arr + 1.0) / arr))
    U = float(np.prod(arr / (arr - 1.0)))
    return L, U, int(arr.size)


def panel1_log_bounds(ax) -> None:
    """Sanity check for log bounds used in the derivation."""
    p = np.logspace(0.2, 4, 250)

    y_lower = 1.0 / (p + 1.0)
    l_lower = np.log((p + 1.0) / p)

    y_upper = 1.0 / (p - 1.0)
    l_upper = np.log(p / (p - 1.0))

    ax.plot(p, l_lower, label=r"$\log((p+1)/p)$")
    ax.plot(p, y_lower, label=r"$1/(p+1)$")
    ax.plot(p, l_upper, label=r"$\log(p/(p-1))$")
    ax.plot(p, y_upper, label=r"$1/(p-1)$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Panel 1: Log-bound sanity checks")
    ax.set_xlabel("p")
    ax.set_ylabel("value")
    ax.legend(frameon=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=200000)
    ap.add_argument("--outdir", default="deliverables")
    ap.add_argument("--dpi", type=int, default=250)
    ap.add_argument("--plot_cap", type=int, default=70000)
    ap.add_argument("--max_k", type=int, default=10)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sig = sigma_sieve(args.N)
    spf = spf_sieve(args.N)

    n_vals = np.arange(2, args.N + 1, dtype=np.int32)

    y = np.empty_like(n_vals, dtype=np.float64)
    L = np.empty_like(n_vals, dtype=np.float64)
    U = np.empty_like(n_vals, dtype=np.float64)
    k = np.empty_like(n_vals, dtype=np.int32)

    for i, n in enumerate(n_vals):
        primes = distinct_primes_from_spf(int(n), spf)
        Li, Ui, ki = LU_from_prime_support(primes)
        L[i], U[i], k[i] = Li, Ui, ki
        y[i] = sig[n] / n

    slack_U = np.maximum(U - y, 1e-16)
    slack_L = np.maximum(y - L, 1e-16)

    fig = plt.figure(figsize=(12, 14), dpi=args.dpi)
    ax1, ax2, ax3 = fig.subplots(3, 1)

    panel1_log_bounds(ax1)

    sel = np.random.default_rng(0).choice(len(n_vals), size=min(len(n_vals), args.plot_cap), replace=False)
    sel.sort()

    ax2.scatter(n_vals[sel], y[sel], s=3, alpha=0.3, label="σ(n)/n")
    ax2.scatter(n_vals[sel], L[sel], s=2, alpha=0.2, label="Lower envelope")
    ax2.scatter(n_vals[sel], U[sel], s=2, alpha=0.2, label="Upper envelope")
    ax2.set_xscale("log")
    ax2.legend(frameon=True)
    ax2.set_title("Panel 2: Prime-support envelope")

    ks = np.arange(1, args.max_k + 1, dtype=int)

    def safe_median(arr: np.ndarray) -> float:
        if arr.size == 0:
            return float("nan")
        return float(np.median(arr))

    for label, data in [("U−y", slack_U), ("y−L", slack_L)]:
        med = []
        for ki in ks:
            subset = data[k == ki]
            med.append(safe_median(subset))
        ax3.plot(ks, med, marker="o", label=label)

    ax3.set_yscale("log")
    ax3.set_xlabel("ω(n)")
    ax3.set_ylabel("gap")
    ax3.legend(frameon=True)
    ax3.set_title("Panel 3: Gap vs prime support")

    fig.tight_layout()
    fig.savefig(outdir / "joiner_inequality_validation_3x1.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
