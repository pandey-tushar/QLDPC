#!/usr/bin/env python3
"""
Surface-code erasure-aware MWPM baseline (code capacity, adaptive threshold search).

Uses per-shot PyMatching weight arrays (erased=0, non-erased=BIG) to correctly
restrict corrections to erased edges only.

ADAPTIVE MODE (default): steps p upward until WER crosses target_wer, then
bisects to find p*. This avoids wasting shots on p-values far from threshold —
identical logic to sweep_scaling.py's adaptive_find_crossing.

Usage:
  python examples/sweep_surface_baseline_erasure_aware.py \\
      --Ls 12,18,24,30,36 --shots 50000 --seed 12345 --tag erasure_aware
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.random import SeedSequence
import pymatching

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qldpc.toric_code import toric_code_matrices


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------

def _erasure_aware_worker(args: Tuple) -> int:
    seed, shots, p_erasure, Hz_dense, Hx_dense, lz_dense, lx_dense, Hz_sparse, Hx_sparse = args
    rng = np.random.default_rng(seed)
    n = Hz_dense.shape[1]
    BIG_WEIGHT = float(n)
    logical_fails = 0
    weights = np.empty(n, dtype=float)

    for _ in range(shots):
        erasure_mask = rng.random(n) < p_erasure
        erased = np.where(erasure_mask)[0]

        x_error = np.zeros(n, dtype=np.uint8)
        z_error = np.zeros(n, dtype=np.uint8)
        if erased.size:
            x_error[erased] = rng.integers(0, 2, size=erased.size, dtype=np.uint8)
            z_error[erased] = rng.integers(0, 2, size=erased.size, dtype=np.uint8)

        weights[:] = BIG_WEIGHT
        if erased.size:
            weights[erased] = 0.0

        m_x = pymatching.Matching(Hz_sparse, weights=weights)
        m_z = pymatching.Matching(Hx_sparse, weights=weights)

        x_syndrome = (Hz_dense @ x_error) % 2
        x_corr = np.asarray(m_x.decode(x_syndrome), dtype=np.uint8)
        x_residual = (x_error + x_corr) % 2
        x_fail = bool(np.any((lz_dense @ x_residual) % 2))

        z_syndrome = (Hx_dense @ z_error) % 2
        z_corr = np.asarray(m_z.decode(z_syndrome), dtype=np.uint8)
        z_residual = (z_error + z_corr) % 2
        z_fail = bool(np.any((lx_dense @ z_residual) % 2))

        if x_fail or z_fail:
            logical_fails += 1

    return logical_fails


# ---------------------------------------------------------------------------
# Single p-value evaluation
# ---------------------------------------------------------------------------

def eval_point(
    L: int,
    p: float,
    shots: int,
    seed: Optional[int],
    num_cores: int,
    Hz_dense, Hx_dense, lz_dense, lx_dense, Hz_sparse, Hx_sparse,
) -> Dict[str, Any]:
    n = Hz_dense.shape[1]
    if seed is None:
        ss = SeedSequence()
    else:
        p_key = int(round(float(p) * 1_000_000_000))
        ss = SeedSequence([int(seed), int(L), p_key])
    children = ss.spawn(num_cores)
    worker_seeds = [int(c.generate_state(1, dtype=np.uint32)[0]) for c in children]

    base = shots // num_cores
    remainder = shots % num_cores
    shot_counts = [base + (1 if i < remainder else 0) for i in range(num_cores)]

    args_list = [
        (worker_seeds[i], shot_counts[i], float(p),
         Hz_dense, Hx_dense, lz_dense, lx_dense, Hz_sparse, Hx_sparse)
        for i in range(num_cores) if shot_counts[i] > 0
    ]

    t0 = time.time()
    with multiprocessing.Pool(num_cores) as pool:
        results = pool.map(_erasure_aware_worker, args_list)
    elapsed = time.time() - t0

    total_fails = int(sum(results))
    actual_shots = int(sum(shot_counts))
    wer = float(total_fails / actual_shots)
    print(f"  p={p:.5f}  WER={wer:.5f}  fails={total_fails}/{actual_shots}  sec={elapsed:.1f}",
          flush=True)
    return {"p": p, "wer": wer, "fails": total_fails, "shots": actual_shots, "seconds": elapsed}


# ---------------------------------------------------------------------------
# Adaptive search: bracket then regula falsi with Illinois modification
# ---------------------------------------------------------------------------

def adaptive_threshold(
    L: int, shots: int, seed: Optional[int], num_cores: int,
    p_start: float, bracket_step: float, p_max: float,
    target_wer: float, tol: float, max_iter: int = 10,
    Hz_dense=None, Hx_dense=None, lz_dense=None, lx_dense=None, Hz_sparse=None, Hx_sparse=None,
) -> Tuple[List[Dict], Optional[float]]:
    """
    Step upward to bracket the WER crossing, then refine using regula falsi
    with the Illinois modification (prevents stagnation when one endpoint
    barely moves). Stops when |p_hi - p_lo| < tol OR after max_iter refinement
    steps (hard cap), whichever comes first.

    Regula falsi step:
        p_next = p_lo + (target - w_lo) / (w_hi - w_lo) * (p_hi - p_lo)

    Illinois modification: if the same endpoint was updated in the previous
    step, halve its effective WER (relative to target) before interpolating,
    forcing the bracket to contract from the stagnant side.
    """
    sampled: List[Dict] = []
    cache: Dict[float, Dict] = {}

    def get(p: float) -> Dict:
        p = round(p, 10)
        if p not in cache:
            r = eval_point(L, p, shots, seed, num_cores,
                           Hz_dense, Hx_dense, lz_dense, lx_dense, Hz_sparse, Hx_sparse)
            cache[p] = r
            sampled.append(r)
        return cache[p]

    # --- Step 1: bracket ---
    p_lo = p_hi = None
    p = p_start
    while p <= p_max + 1e-12:
        r = get(p)
        if r["wer"] < target_wer:
            p_lo = p
            p = round(p + bracket_step, 10)
        else:
            p_hi = p
            break

    if p_lo is None or p_hi is None:
        return sampled, None

    # --- Step 2: regula falsi with Illinois modification ---
    last_side = None   # tracks which endpoint was updated last iteration
    w_lo = cache[round(p_lo, 10)]["wer"]
    w_hi = cache[round(p_hi, 10)]["wer"]

    for _iter in range(max_iter - len(sampled)):
        if (p_hi - p_lo) <= tol:
            break
        # Illinois: if the same side was updated twice in a row,
        # halve the distance of that endpoint's WER from target_wer
        # so the interpolation is pushed toward the stagnant side.
        eff_w_lo = w_lo
        eff_w_hi = w_hi
        if last_side == "lo":
            eff_w_hi = target_wer + (w_hi - target_wer) / 2.0
        elif last_side == "hi":
            eff_w_lo = target_wer - (target_wer - w_lo) / 2.0

        denom = eff_w_hi - eff_w_lo
        if abs(denom) < 1e-15:
            # Flat segment — fall back to midpoint
            p_next = 0.5 * (p_lo + p_hi)
        else:
            t = (target_wer - eff_w_lo) / denom
            # Clamp t to (0.05, 0.95) to avoid shooting to boundary
            t = max(0.05, min(0.95, t))
            p_next = p_lo + t * (p_hi - p_lo)

        p_next = round(p_next, 10)
        r = get(p_next)

        # Stop early if this point is already close enough to target WER
        if abs(r["wer"] - target_wer) <= tol * target_wer:
            p_lo = p_hi = p_next
            break

        if r["wer"] < target_wer:
            p_lo = p_next
            w_lo = r["wer"]
            last_side = "lo"
        else:
            p_hi = p_next
            w_hi = r["wer"]
            last_side = "hi"

    # Final linear interpolation on the tight bracket
    if abs(w_hi - w_lo) < 1e-15:
        p_star = 0.5 * (p_lo + p_hi)
    else:
        t = (target_wer - w_lo) / (w_hi - w_lo)
        p_star = p_lo + t * (p_hi - p_lo)

    return sampled, p_star


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_pkg_version(name: str) -> Optional[str]:
    try:
        from importlib import metadata
        return metadata.version(name)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Erasure-aware toric code MWPM baseline — adaptive threshold search."
    )
    parser.add_argument("--Ls", type=str, default="12,18,24,30,36")
    parser.add_argument("-s", "--shots", type=int, default=50000)
    parser.add_argument("--p-start",      type=float, default=0.38,
                        help="Starting erasure rate for upward step (default 0.38)")
    parser.add_argument("--bracket-step", type=float, default=0.04,
                        help="Step size while searching for crossing (default 0.04)")
    parser.add_argument("--p-max",        type=float, default=0.60,
                        help="Maximum p to try (default 0.60)")
    parser.add_argument("--target-wer",   type=float, default=0.10)
    parser.add_argument("--tol", type=float, default=1e-4,
                        help="Stop when bracket width < tol (default 1e-4)")
    parser.add_argument("--max-iter", type=int, default=10,
                        help="Hard cap on regula falsi refinement iterations (default 10)")
    parser.add_argument("--seed",  type=int,  default=None)
    parser.add_argument("--cores", type=int,  default=None)
    parser.add_argument("--tag",   type=str,  default="surface_mwpm_erasure_aware")
    args = parser.parse_args()

    num_cores = args.cores or max(1, multiprocessing.cpu_count() - 1)
    Ls = [int(x.strip()) for x in args.Ls.split(",") if x.strip()]

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path  = out_dir / f"baseline_surface_{stamp}_{args.tag}.csv"
    json_path = out_dir / f"baseline_surface_{stamp}_{args.tag}.json"

    rows: List[Dict] = []
    threshold_estimates: List[Dict] = []

    for L in Ls:
        N = 2 * L * L
        print(f"\n=== L={L}  N={N}  K=2  shots={args.shots}  cores={num_cores} ===")

        Hx, Hz, lz, lx = toric_code_matrices(L)
        Hz_dense = np.asarray(Hz.toarray(), dtype=np.uint8)
        Hx_dense = np.asarray(Hx.toarray(), dtype=np.uint8)
        lz_dense = np.asarray(lz, dtype=np.uint8)
        lx_dense = np.asarray(lx, dtype=np.uint8)

        sampled, p_star = adaptive_threshold(
            L=L, shots=args.shots, seed=args.seed, num_cores=num_cores,
            p_start=args.p_start, bracket_step=args.bracket_step,
            p_max=args.p_max, target_wer=args.target_wer,
            tol=args.tol, max_iter=args.max_iter,
            Hz_dense=Hz_dense, Hx_dense=Hx_dense,
            lz_dense=lz_dense, lx_dense=lx_dense,
            Hz_sparse=Hz, Hx_sparse=Hx,
        )

        p_str = f"{p_star:.5f}" if p_star is not None else "not found"
        print(f"  --> p* ~ {p_str}  ({len(sampled)} p-values evaluated)")

        for r in sampled:
            rows.append({
                "family": "toric_surface_mwpm_erasure_aware",
                "L": L, "M": L, "N": N, "K": 2,
                "p":       r["p"],
                "shots":   r["shots"],
                "fails":   r["fails"],
                "wer":     r["wer"],
                "seconds": r["seconds"],
            })
        threshold_estimates.append({
            "L": L, "M": L, "N": N, "K": 2,
            "target_wer": args.target_wer,
            "p_at_target_wer": p_star,
            "n_points_evaluated": len(sampled),
        })

    fieldnames = ["family", "L", "M", "N", "K", "p", "shots", "fails", "wer", "seconds"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    meta = {
        "timestamp_utc": stamp,
        "tag": args.tag,
        "family": "toric_surface_mwpm_erasure_aware",
        "decoder": "pymatching_per_shot_weights_adaptive",
        "note": "Adaptive: bracket then regula falsi (Illinois). Per-shot weights: erased=0, non-erased=N.",
        "shots_per_point": args.shots,
        "num_cores": num_cores,
        "p_start": args.p_start,
        "bracket_step": args.bracket_step,
        "tol": args.tol,
        "target_wer": args.target_wer,
        "seed": args.seed,
        "threshold_estimates": threshold_estimates,
        "platform": {"system": platform.system(), "release": platform.release()},
        "python": {"version": platform.python_version()},
        "package_versions": {
            "numpy": safe_pkg_version("numpy"),
            "pymatching": safe_pkg_version("pymatching"),
        },
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    print(f"\nSaved CSV : {csv_path}")
    print(f"Saved JSON: {json_path}")
    print("\nThreshold summary:")
    for e in threshold_estimates:
        p_str = f"{e['p_at_target_wer']:.5f}" if e["p_at_target_wer"] else "not found"
        print(f"  L={e['L']}  N={e['N']}  p* ~ {p_str}  ({e['n_points_evaluated']} evals)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
