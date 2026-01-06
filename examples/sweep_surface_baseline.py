#!/usr/bin/env python3
"""
Surface-code MWPM baseline (code capacity).

IMPORTANT: With erasure conversion, the decoder would ideally use the per-shot erasure locations.
PyMatching does not support per-shot weight updates efficiently, so this baseline uses an
"uninformed" MWPM model:
  - Erasure rate p is converted to an effective X error rate q = p/2
  - MWPM is run with constant weights derived from q

This is still a useful baseline for comparison, but it is conservative for surface codes
under true erasure-informed decoding.

Outputs CSV/JSON into results/ with a schema compatible with plot_results.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pymatching

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qldpc.toric_code import toric_code_matrices


def frange(start: float, stop: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("--p-step must be > 0")
    vals = []
    x = start
    while x <= stop + 1e-12:
        vals.append(round(x, 10))
        x += step
    return vals


def estimate_p_at_target(points: List[Tuple[float, float]], target: float) -> float | None:
    pts = sorted(points, key=lambda t: t[0])
    for (p0, w0), (p1, w1) in zip(pts, pts[1:]):
        if (w0 - target) * (w1 - target) <= 0 and w0 != w1:
            t = (target - w0) / (w1 - w0)
            return p0 + t * (p1 - p0)
    return None


def run_mwpm_point(L: int, p_erasure: float, shots: int, seed: int = 0) -> Dict[str, Any]:
    """
    Toric code MWPM baseline for X-errors only (decode using plaquette checks Hz).
    """
    rng = np.random.default_rng(seed)

    Hx, Hz, lz = toric_code_matrices(L)
    n = Hz.shape[1]

    # Erasure conversion: X error occurs with probability q = p/2 (marginally).
    q = float(p_erasure) / 2.0
    q = min(max(q, 1e-12), 1 - 1e-12)

    # MWPM edge weights (negative log-likelihood ratio up to constant)
    w = math.log((1.0 - q) / q)
    weights = np.full(n, w, dtype=float)
    matching = pymatching.Matching(Hz, weights=weights)

    fails = 0
    for _ in range(shots):
        error = (rng.random(n) < q).astype(np.uint8)
        syndrome = (Hz @ error) % 2
        corr = matching.decode(syndrome)
        if not isinstance(corr, np.ndarray):
            corr = np.array(corr, dtype=np.uint8)
        else:
            corr = corr.astype(np.uint8, copy=False)
        residual = (error + corr) % 2

        # Word failure if residual anticommutes with any logical Z
        if np.any((lz @ residual) % 2):
            fails += 1

    wer = fails / shots
    return {"wer": float(wer), "fails": int(fails), "shots": int(shots)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Surface code MWPM baseline (toric) sweep.")
    parser.add_argument("--Ls", type=str, default="12,18,24,30,36", help="Comma-separated toric sizes L (default: 12,18,24,30,36)")
    parser.add_argument("-s", "--shots", type=int, default=20000, help="Shots per p per L (default: 20000)")
    parser.add_argument("--p-min", type=float, default=0.30)
    parser.add_argument("--p-max", type=float, default=0.60)
    parser.add_argument("--p-step", type=float, default=0.02)
    parser.add_argument("--target-wer", type=float, default=0.10)
    parser.add_argument("--tag", type=str, default="surface_mwpm", help="Tag for output filenames")
    args = parser.parse_args()

    Ls = [int(x.strip()) for x in args.Ls.split(",") if x.strip()]
    ps = frange(args.p_min, args.p_max, args.p_step)

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = out_dir / f"baseline_surface_{stamp}_{args.tag}.csv"
    json_path = out_dir / f"baseline_surface_{stamp}_{args.tag}.json"

    rows = []
    threshold_estimates = []

    for L in Ls:
        pts = []
        N = 2 * L * L
        print(f"\n=== TORIC SURFACE BASELINE === L={L}  N={N}  shots={args.shots}")
        for p in ps:
            start = datetime.now()
            r = run_mwpm_point(L=L, p_erasure=float(p), shots=int(args.shots), seed=np.random.randint(1_000_000_000))
            sec = (datetime.now() - start).total_seconds()
            print(f"  p={p:.4f}  WER={r['wer']:.5f}  fails={r['fails']}/{r['shots']}  sec={sec:.2f}")
            pts.append((float(p), float(r["wer"])))
            rows.append(
                {
                    "family": "toric_surface_mwpm_uninformed",
                    "L": int(L),
                    "M": int(L),  # keep schema consistent (use M=L)
                    "N": int(N),
                    "K": 2,
                    "p": float(p),
                    "shots": int(r["shots"]),
                    "fails": int(r["fails"]),
                    "wer": float(r["wer"]),
                    "seconds": float(sec),
                }
            )

        p_star = estimate_p_at_target(pts, float(args.target_wer))
        threshold_estimates.append(
            {"L": int(L), "M": int(L), "N": int(N), "K": 2, "target_wer": float(args.target_wer), "p_at_target_wer": p_star}
        )

    # Write CSV
    fieldnames = ["family", "L", "M", "N", "K", "p", "shots", "fails", "wer", "seconds"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    meta = {
        "timestamp_utc": stamp,
        "tag": args.tag,
        "family": "toric_surface_mwpm_uninformed",
        "shots_per_point": int(args.shots),
        "p_grid": ps,
        "Ls": Ls,
        "target_wer": float(args.target_wer),
        "threshold_estimates": threshold_estimates,
        "note": "This baseline uses MWPM with q=p/2 and does not condition on per-shot erasure locations.",
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")
    print("Approx p@target_wer:")
    for e in threshold_estimates:
        print(f"  L={e['L']} N={e['N']}: p@WER={e['target_wer']:.3f} ≈ {e['p_at_target_wer']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


