#!/usr/bin/env python3
"""
Scaling sweep runner (code capacity, erasure channel).

Runs multiple Bivariate Bicycle code sizes, sweeps erasure rates, and saves:
- results CSV: one row per (L, M, p)
- metadata JSON: run configuration + environment info

Example:
  python examples/sweep_scaling.py -s 200000 --sizes 10x6,12x6,14x6 --p-min 0.30 --p-max 0.42 --p-step 0.01
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qldpc import BivariateBicycleCode, QuantumSimulator, DecoderConfig


def parse_sizes(text: str) -> List[Tuple[int, int]]:
    """
    Parse sizes like: "12x6,14x6" -> [(12,6),(14,6)]
    """
    out: List[Tuple[int, int]] = []
    for part in text.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if "x" not in part:
            raise ValueError(f"Bad size '{part}'. Use e.g. 12x6")
        a, b = part.split("x", 1)
        out.append((int(a), int(b)))
    if not out:
        raise ValueError("No sizes provided")
    return out


def frange(start: float, stop: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("--p-step must be > 0")
    vals = []
    x = start
    while x <= stop + 1e-12:
        vals.append(round(x, 10))
        x += step
    return vals


def estimate_p_at_target_wer(points: List[Tuple[float, float]], target_wer: float) -> float | None:
    """
    Given (p, wer) sorted by p, return interpolated p where wer crosses target_wer.
    Returns None if it never crosses within range.
    """
    pts = sorted(points, key=lambda t: t[0])
    for (p0, w0), (p1, w1) in zip(pts, pts[1:]):
        if (w0 - target_wer) == 0:
            return p0
        # crossing between w0 and w1 (inclusive)
        if (w0 - target_wer) * (w1 - target_wer) <= 0 and w0 != w1:
            t = (target_wer - w0) / (w1 - w0)
            return p0 + t * (p1 - p0)
    return None


def adaptive_find_crossing(
    sim: QuantumSimulator,
    target_wer: float,
    shots: int,
    p_start: float,
    p_step: float,
    p_max: float,
    refine_steps: int,
    pool,
    prepared: bool,
):
    """
    Bracket the crossing (WER crosses target_wer) by stepping p upward,
    then refine by bisection.

    Returns:
      - sampled_points: List[(p, wer, fails, shots, seconds)]
      - p_at_target: float | None
    """
    sampled = []
    cache = {}

    def eval_p(p: float):
        p = float(round(p, 10))
        if p in cache:
            return cache[p]
        r = sim.run_point_prepared(p, shots, pool=pool) if prepared else sim.run_point(p, shots, pool=pool)
        row = (p, r["wer"], int(r["fails"]), int(r["shots"]), float(r["seconds"]))
        cache[p] = row
        sampled.append(row)
        # Progress line (adaptive mode can have long per-point runtimes)
        print(f"  p={p:.5f}  wer={row[1]:.5f}  fails={row[2]}  shots={row[3]}  sec={row[4]:.2f}", flush=True)
        return row

    # Bracket
    p_lo = None
    w_lo = None
    p = p_start
    while p <= p_max + 1e-12:
        p_i, w_i, *_ = eval_p(p)
        if w_i < target_wer:
            p_lo, w_lo = p_i, w_i
            p += p_step
            continue
        # first above target
        if p_lo is None:
            # We started above target; can't bracket below with this start
            return sampled, None
        p_hi, w_hi = p_i, w_i

        # Refine by bisection
        for _ in range(refine_steps):
            mid = 0.5 * (p_lo + p_hi)
            p_m, w_m, *_ = eval_p(mid)
            if w_m < target_wer:
                p_lo, w_lo = p_m, w_m
            else:
                p_hi, w_hi = p_m, w_m

        # Linear interpolation on last bracket
        if w_hi == w_lo:
            return sampled, float(p_hi)
        t = (target_wer - w_lo) / (w_hi - w_lo)
        return sampled, float(p_lo + t * (p_hi - p_lo))

    return sampled, None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run scaling sweeps and save CSV/JSON results.")
    parser.add_argument(
        "--sizes",
        type=parse_sizes,
        default=parse_sizes("12x6"),
        help="Comma-separated code sizes LxM, e.g. --sizes 10x6,12x6,14x6",
    )
    parser.add_argument("-s", "--shots", type=int, default=200000, help="Shots (used for each evaluated p)")
    parser.add_argument("--p-min", type=float, default=0.30, help="Min erasure rate (non-adaptive mode)")
    parser.add_argument("--p-max", type=float, default=0.42, help="Max erasure rate (also adaptive max)")
    parser.add_argument("--p-step", type=float, default=0.01, help="Erasure rate step (non-adaptive mode)")
    parser.add_argument("--cores", type=int, default=None, help="Override core count (default: cpu_count()-1)")

    # Decoder knobs
    parser.add_argument("--bp-method", type=str, default="min_sum")
    parser.add_argument("--osd-method", type=str, default="osd_cs")
    parser.add_argument("--osd-order", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=50)

    # Output
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results",
        help="Output directory (default: results)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional run tag appended to filenames, e.g. --tag laptop1",
    )
    parser.add_argument(
        "--target-wer",
        type=float,
        default=0.10,
        help="Also compute an approximate p where WER crosses this value (default 0.10)",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Adaptive sweep: step p upward until WER crosses target, then bisection-refine.",
    )
    parser.add_argument(
        "--p-start",
        type=float,
        default=0.30,
        help="Adaptive mode: starting p (default 0.30)",
    )
    parser.add_argument(
        "--bracket-step",
        type=float,
        default=0.02,
        help="Adaptive mode: step size while searching for crossing (default 0.02)",
    )
    parser.add_argument(
        "--refine-steps",
        type=int,
        default=6,
        help="Adaptive mode: number of bisection refinements after bracketing (default 6)",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = f"_{args.tag}" if args.tag else ""
    csv_path = out_dir / f"scaling_{stamp}{tag}.csv"
    meta_path = out_dir / f"scaling_{stamp}{tag}.json"

    erasure_rates = frange(args.p_min, args.p_max, args.p_step) if not args.adaptive else []

    config = DecoderConfig(
        bp_method=args.bp_method,
        osd_method=args.osd_method,
        osd_order=args.osd_order,
        max_iter=args.max_iter,
    )

    # Metadata
    meta = {
        "timestamp_utc": stamp,
        "tag": args.tag,
        "shots_per_point": args.shots,
        "erasure_rates": erasure_rates,
        "sizes": [{"L": L, "M": M, "N": 2 * L * M} for (L, M) in args.sizes],
        "decoder_config": asdict(config),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
    }

    rows = []

    for (L, M) in args.sizes:
        code = BivariateBicycleCode(L=L, M=M)
        sim = QuantumSimulator(code, config=config, num_cores=args.cores)

        # We can infer K (encoded qubits) by creating the decoder once (cheap compared to full sweep)
        # but keep it optional to avoid hard-failing if upstream API changes.
        try:
            from qldpc.decoder import create_decoder

            qcode, _ = create_decoder(sim.Hx, sim.Hz, config, initial_error_rate=0.1)
            K = getattr(qcode, "K", None)
        except Exception:
            K = None

        print("\n=== SIZE INFO ===")
        print(f"L={L} M={M} N={2*L*M}  Hx={sim.Hx.shape} Hz={sim.Hz.shape}  K={K}")
        if K == 0:
            print("Skipping this size because K=0 (encodes no logical qubits), so WER is not meaningful.")
            meta.setdefault("skipped_sizes", []).append(
                {"L": L, "M": M, "N": int(2 * L * M), "reason": "K=0"}
            )
            continue
        if K is None:
            print("Warning: Could not determine K for this size. Proceeding anyway (but interpret results carefully).")

        pts = []
        if args.adaptive:
            # Prepared pool per size: avoids recomputing logicals/decoder per p
            pool, info = sim.make_prepared_pool()
            if K is None:
                K = info.get("K")
            sampled, p_star = adaptive_find_crossing(
                sim=sim,
                target_wer=float(args.target_wer),
                shots=int(args.shots),
                p_start=float(args.p_start),
                p_step=float(args.bracket_step),
                p_max=float(args.p_max),
                refine_steps=int(args.refine_steps),
                pool=pool,
                prepared=True,
            )
            pool.close()
            pool.join()
            for (p, wer, fails, shots, seconds) in sampled:
                pts.append((p, wer))
                rows.append(
                    {
                        "family": "bivariate_bicycle",
                        "L": L,
                        "M": M,
                        "N": int(2 * L * M),
                        "K": "" if K is None else int(K),
                        "p": float(p),
                        "shots": int(shots),
                        "fails": int(fails),
                        "wer": float(wer),
                        "seconds": float(seconds),
                    }
                )
        else:
            # Non-adaptive mode: use existing implementation (one pool per experiment)
            results = sim.run_experiment(
                erasure_rates=erasure_rates,
                total_shots=args.shots,
                verbose=True,
                return_details=True,
            )

            for p in erasure_rates:
                r = results[p]
                pts.append((p, r["wer"]))
                rows.append(
                    {
                        "family": "bivariate_bicycle",
                        "L": L,
                        "M": M,
                        "N": int(2 * L * M),
                        "K": "" if K is None else int(K),
                        "p": float(p),
                        "shots": int(r["shots"]),
                        "fails": int(r["fails"]),
                        "wer": float(r["wer"]),
                        "seconds": float(r["seconds"]),
                    }
                )

            p_star = estimate_p_at_target_wer(pts, float(args.target_wer))
        meta.setdefault("threshold_estimates", []).append(
            {
                "L": L,
                "M": M,
                "N": int(2 * L * M),
                "K": None if K is None else int(K),
                "target_wer": float(args.target_wer),
                "p_at_target_wer": None if p_star is None else float(p_star),
            }
        )

    # Write CSV
    fieldnames = ["family", "L", "M", "N", "K", "p", "shots", "fails", "wer", "seconds"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    # Write metadata JSON
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved JSON: {meta_path}")
    if "threshold_estimates" in meta:
        print("\nApprox p@target_wer:")
        for item in meta["threshold_estimates"]:
            print(
                f"  L={item['L']} M={item['M']} N={item['N']}: "
                f"p@WER={item['target_wer']:.3f} ≈ {item['p_at_target_wer']}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


