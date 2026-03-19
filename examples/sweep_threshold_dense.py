#!/usr/bin/env python3
"""
Dense near-threshold sweep for finite-size scaling (FSS) analysis.

This script runs a fine-grained p-grid around each code size's known pseudo-threshold,
collecting high-statistics WER data needed for:
  1. Accurate FSS collapse plots (WER vs scaled variable)
  2. Bootstrap confidence intervals on p*
  3. Proper error bars in all figures

Uses the existing pseudo-threshold estimates (from JSON files) to center the window
around each code's crossing point, then sweeps with small step size and high shot count.

Usage (recommended for journal):
  python examples/sweep_threshold_dense.py \\
      --sizes 12x6,18x9,24x12,30x15,36x18 \\
      --shots 500000          \\
      --p-half-window 0.06    \\
      --p-step 0.005          \\
      --seed 12345            \\
      --tag dense_fss

  # For even better statistics near the largest codes:
  python examples/sweep_threshold_dense.py \\
      --sizes 30x15,36x18    \\
      --shots 1000000        \\
      --p-half-window 0.04   \\
      --p-step 0.004         \\
      --seed 12345            \\
      --tag dense_fss_highstats

The known pseudo-threshold estimates (from existing results) are used as defaults
but can be overridden via --p-centers.

Outputs: results/scaling_<timestamp>_<tag>.csv  and  .json
(same schema as sweep_scaling.py → compatible with finite_size_scaling_analysis.py)
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qldpc import BivariateBicycleCode, QuantumSimulator, DecoderConfig


# Known pseudo-thresholds from existing runs (p* at WER=0.10)
# Update these from your JSON results if you run new sweeps first.
KNOWN_P_STARS: Dict[Tuple[int, int], float] = {
    (12,  6): 0.370,
    (18,  9): 0.439,
    (24, 12): 0.445,
    (30, 15): 0.467,
    (36, 18): 0.471,
}


def parse_sizes(text: str) -> List[Tuple[int, int]]:
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


def parse_centers(text: str) -> Dict[str, float]:
    """Parse '--p-centers 12x6:0.370,18x9:0.439' into {(L,M): p*}."""
    result = {}
    if not text:
        return result
    for item in text.split(","):
        item = item.strip()
        if ":" not in item:
            continue
        size_str, val_str = item.rsplit(":", 1)
        result[size_str.strip()] = float(val_str)
    return result


def frange_centered(center: float, half_width: float, step: float) -> List[float]:
    start = round(center - half_width, 10)
    stop  = round(center + half_width, 10)
    vals: List[float] = []
    x = start
    while x <= stop + 1e-12:
        vals.append(round(x, 10))
        x += step
    return vals


def estimate_p_at_target_wer(
    points: List[Tuple[float, float]], target_wer: float
) -> Optional[float]:
    pts = sorted(points, key=lambda t: t[0])
    for (p0, w0), (p1, w1) in zip(pts, pts[1:]):
        if (w0 - target_wer) * (w1 - target_wer) <= 0 and w0 != w1:
            t = (target_wer - w0) / (w1 - w0)
            return p0 + t * (p1 - p0)
    return None


def safe_pkg_version(name: str) -> Optional[str]:
    try:
        from importlib import metadata
        return metadata.version(name)
    except Exception:
        try:
            import pkg_resources
            return pkg_resources.get_distribution(name).version
        except Exception:
            return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dense near-threshold sweep for FSS analysis."
    )
    parser.add_argument(
        "--sizes", type=parse_sizes, default=parse_sizes("12x6,18x9,24x12,30x15,36x18"),
        help="Comma-separated code sizes LxM (default: all 5 standard sizes)",
    )
    parser.add_argument(
        "-s", "--shots", type=int, default=500000,
        help="Shots per (L, M, p) point (default: 500000)",
    )
    parser.add_argument(
        "--p-half-window", type=float, default=0.06,
        help="Half-width around each size's p* (default: 0.06 → covers p* ± 0.06)",
    )
    parser.add_argument(
        "--p-step", type=float, default=0.005,
        help="Step size within the window (default: 0.005)",
    )
    parser.add_argument(
        "--p-centers", type=str, default="",
        help="Override p* per size: '12x6:0.370,18x9:0.439' (default: use built-in estimates)",
    )
    parser.add_argument(
        "--target-wer", type=float, default=0.10,
    )
    parser.add_argument(
        "--seed", type=int, default=None,
    )
    parser.add_argument(
        "--tag", type=str, default="dense_fss",
    )
    parser.add_argument(
        "--out-dir", type=str, default="results",
    )
    parser.add_argument("--cores", type=int, default=None)
    # Decoder knobs (keep same defaults as sweep_scaling.py)
    parser.add_argument("--bp-method",  type=str, default="min_sum")
    parser.add_argument("--osd-method", type=str, default="osd_cs")
    parser.add_argument("--osd-order",  type=int, default=10)
    parser.add_argument("--max-iter",   type=int, default=50)
    args = parser.parse_args()

    # Build p-center overrides
    custom_centers = parse_centers(args.p_centers)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = f"_{args.tag}" if args.tag else ""
    csv_path  = out_dir / f"scaling_{stamp}{tag}.csv"
    meta_path = out_dir / f"scaling_{stamp}{tag}.json"

    config = DecoderConfig(
        bp_method=args.bp_method,
        osd_method=args.osd_method,
        osd_order=args.osd_order,
        max_iter=args.max_iter,
    )

    meta = {
        "timestamp_utc": stamp,
        "tag": args.tag,
        "purpose": "dense_near_threshold_fss",
        "shots_per_point": args.shots,
        "p_half_window": args.p_half_window,
        "p_step": args.p_step,
        "target_wer": args.target_wer,
        "sizes": [{"L": L, "M": M, "N": 2 * L * M} for (L, M) in args.sizes],
        "decoder_config": asdict(config),
        "seed": args.seed,
        "python": {"version": platform.python_version()},
        "platform": {"system": platform.system(), "release": platform.release()},
        "package_versions": {
            "numpy":  safe_pkg_version("numpy"),
            "scipy":  safe_pkg_version("scipy"),
            "ldpc":   safe_pkg_version("ldpc"),
            "bposd":  safe_pkg_version("bposd"),
        },
    }

    rows: List[Dict] = []
    seed_map: List[Dict] = []

    for (L, M) in args.sizes:
        N = 2 * L * M
        size_str = f"{L}x{M}"

        # Determine p-center
        if size_str in custom_centers:
            p_center = custom_centers[size_str]
        elif (L, M) in KNOWN_P_STARS:
            p_center = KNOWN_P_STARS[(L, M)]
        else:
            print(f"[WARN] No known p* for {size_str}. Using p_center=0.45.")
            p_center = 0.45

        erasure_rates = frange_centered(p_center, args.p_half_window, args.p_step)
        # Clip to [0.01, 0.98]
        erasure_rates = [p for p in erasure_rates if 0.01 <= p <= 0.98]

        print(f"\n=== DENSE FSS SWEEP === {size_str}  N={N}  p*≈{p_center:.3f}")
        print(f"  p range: [{erasure_rates[0]:.3f}, {erasure_rates[-1]:.3f}]  step={args.p_step}  points={len(erasure_rates)}")
        print(f"  shots/point: {args.shots:,}")

        code = BivariateBicycleCode(L=L, M=M)
        sim = QuantumSimulator(code, config=config, num_cores=args.cores, seed=args.seed)

        try:
            from qldpc.decoder import create_decoder
            qcode, _ = create_decoder(sim.Hx, sim.Hz, config, initial_error_rate=0.1)
            K = getattr(qcode, "K", None)
        except Exception:
            K = None

        print(f"  Hx={sim.Hx.shape}  Hz={sim.Hz.shape}  K={K}")
        if K == 0:
            print(f"  Skipping: K=0 (trivial code)")
            continue

        # Use prepared pool for efficiency (decoder initialized once per worker)
        pool, info = sim.make_prepared_pool()
        if K is None:
            K = info.get("K")

        pts: List[Tuple[float, float]] = []
        for p in erasure_rates:
            r = sim.run_point_prepared(p, args.shots, pool=pool, seed=args.seed)
        print(
            f"  p={p:.4f}  WER={r['wer']:.5f}  fails={r['fails']:>7}/{r['shots']}  sec={r['seconds']:.1f}",
                flush=True,
            )
            pts.append((p, r["wer"]))
            rows.append(
                {
                    "family": "bivariate_bicycle",
                    "L": L, "M": M, "N": int(2 * L * M),
                    "K": "" if K is None else int(K),
                    "p": float(p),
                    "shots": int(r["shots"]),
                    "fails": int(r["fails"]),
                    "wer":   float(r["wer"]),
                    "seconds": float(r["seconds"]),
                }
            )
            if r.get("worker_seeds"):
                seed_map.append({"L": int(L), "M": int(M), "p": float(p),
                                 "worker_seeds": list(r["worker_seeds"])})

        pool.close()
        pool.join()

        p_star_new = estimate_p_at_target_wer(pts, args.target_wer)
        meta.setdefault("threshold_estimates", []).append(
            {
                "L": L, "M": M, "N": int(2 * L * M),
                "K": None if K is None else int(K),
                "p_center_used": float(p_center),
                "target_wer": float(args.target_wer),
                "p_at_target_wer": None if p_star_new is None else float(p_star_new),
            }
        )

    # Write CSV
    fieldnames = ["family", "L", "M", "N", "K", "p", "shots", "fails", "wer", "seconds"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    if seed_map:
        meta["seed_map"] = seed_map
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    print(f"\nSaved CSV : {csv_path}")
    print(f"Saved JSON: {meta_path}")
    if "threshold_estimates" in meta:
        print("\nUpdated pseudo-thresholds from dense sweep:")
        for item in meta["threshold_estimates"]:
            p_str = f"{item['p_at_target_wer']:.5f}" if item["p_at_target_wer"] is not None else "not found"
            print(f"  L={item['L']} M={item['M']} N={item['N']}  p* ~ {p_str}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
