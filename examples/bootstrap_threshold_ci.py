#!/usr/bin/env python3
"""
Bootstrap confidence intervals for pseudo-threshold (WER crossing point) estimates.

Addresses Reviewer Issue #3 ("Insufficient statistical rigor in threshold extraction"):
  - Computes 95% bootstrap CIs on each size's p* via parametric resampling
  - Justifies 200k shot count using a shot-count sensitivity analysis
  - Reports binomial standard errors on raw WER points
  - Produces a summary table suitable for the paper

Usage:
  python examples/bootstrap_threshold_ci.py \\
      --csv results/scaling_*.csv \\
      --bootstrap 5000 \\
      --target-wer 0.10 \\
      --out-dir results/ci

Outputs:
  results/ci/threshold_ci_table.json    -- machine-readable summary
  results/ci/threshold_ci_table.txt     -- human-readable table for paper
  results/ci/shot_sensitivity.pdf       -- how p* CI width changes with N_shots
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.stats as scs

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(csv_paths: List[str]) -> Dict[Tuple[int, int], List[Tuple[float, int, int]]]:
    """Load BB code CSV data → {(L,M): [(p, fails, shots), ...]} sorted by p."""
    data: Dict[Tuple[int, int], List] = {}
    for path in csv_paths:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("family", "") != "bivariate_bicycle":
                    continue
                try:
                    L = int(row["L"]); M = int(row["M"])
                    p = float(row["p"])
                    fails = int(row["fails"]); shots = int(row["shots"])
                except (KeyError, ValueError):
                    continue
                data.setdefault((L, M), []).append((p, fails, shots))
    for key in data:
        data[key].sort(key=lambda t: t[0])
    return data


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------

def interpolate_crossing(
    pts: List[Tuple[float, float]], target: float
) -> Optional[float]:
    """Linear interpolation for WER crossing point."""
    for i in range(len(pts) - 1):
        p0, w0 = pts[i]; p1, w1 = pts[i + 1]
        if (w0 - target) * (w1 - target) <= 0 and w0 != w1:
            t = (target - w0) / (w1 - w0)
            return p0 + t * (p1 - p0)
    return None


def bootstrap_threshold_ci(
    raw_pts: List[Tuple[float, int, int]],
    target_wer: float,
    n_boot: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float, float, float]:
    """
    Parametric bootstrap CI for p* using per-point binomial resampling.

    For each bootstrap iteration:
      - At each (p, fails, shots) point, resample fails_b ~ Binomial(shots, fails/shots)
      - Recompute p* by linear interpolation
    Returns: (p_star_obs, mean_b, std_b, ci_lo, ci_hi)
    """
    wer_pts = [(p, f / s) for p, f, s in raw_pts]
    p_star_obs = interpolate_crossing(wer_pts, target_wer)
    if p_star_obs is None:
        return (float("nan"),) * 5

    boot_stars: List[float] = []
    for _ in range(n_boot):
        boot_wer = []
        for p_val, f, s in raw_pts:
            p_hat = f / s
            f_b = int(rng.binomial(s, p_hat))
            boot_wer.append((p_val, f_b / s))
        ps_b = interpolate_crossing(boot_wer, target_wer)
        if ps_b is not None:
            boot_stars.append(ps_b)

    boot_arr = np.array(boot_stars)
    ci_lo = float(np.percentile(boot_arr, 2.5))
    ci_hi = float(np.percentile(boot_arr, 97.5))
    return (
        float(p_star_obs),
        float(np.mean(boot_arr)),
        float(np.std(boot_arr)),
        ci_lo,
        ci_hi,
    )


def wilson_ci(fails: int, shots: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score CI for WER = fails/shots."""
    p_hat = fails / shots
    denom = 1 + z**2 / shots
    center = (p_hat + z**2 / (2 * shots)) / denom
    half = z * np.sqrt(p_hat * (1 - p_hat) / shots + z**2 / (4 * shots**2)) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


# ---------------------------------------------------------------------------
# Shot-count sensitivity analysis
# ---------------------------------------------------------------------------

def shot_sensitivity(
    raw_pts: List[Tuple[float, int, int]],
    target_wer: float,
    shot_counts: List[int],
    n_boot: int,
    rng: np.random.Generator,
) -> Dict[int, float]:
    """
    For each hypothetical N_shots, sub-sample each point's shot count
    and estimate the CI half-width via bootstrap.
    Returns: {N_shots: ci_half_width}
    """
    results = {}
    actual_shots = min(s for _, _, s in raw_pts)
    for n_s in shot_counts:
        if n_s > actual_shots:
            results[n_s] = float("nan")
            continue
        sub_pts = [(p, max(0, int(rng.binomial(n_s, f / s))), n_s) for p, f, s in raw_pts]
        _, _, _, lo, hi = bootstrap_threshold_ci(sub_pts, target_wer, n_boot // 5, rng)
        results[n_s] = float((hi - lo) / 2) if not np.isnan(lo) else float("nan")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bootstrap CIs for pseudo-threshold estimates from sweep_scaling.py CSVs."
    )
    parser.add_argument("--csv", nargs="+", required=True)
    parser.add_argument("--target-wer", type=float, default=0.10)
    parser.add_argument(
        "--bootstrap", type=int, default=5000,
        help="Bootstrap iterations per code size (default: 5000)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="results/ci")
    parser.add_argument(
        "--shot-sensitivity", action="store_true",
        help="Also run shot-count sensitivity analysis (slow)",
    )
    args = parser.parse_args()

    csv_paths: List[str] = []
    for pattern in args.csv:
        expanded = glob.glob(pattern)
        csv_paths.extend(expanded if expanded else [pattern])

    data = load_data(csv_paths)
    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_json = []
    table_lines = [
        f"{'Code (L,M)':<14} {'N':>6} {'K est.':>7} {'p*':>8} {'CI lo':>8} {'CI hi':>8} {'±(half)':>8} {'shots/pt':>10} {'boot_std':>10}",
        "-" * 80,
    ]

    for key in sorted(data):
        L, M = key
        N = 2 * L * M
        pts = data[key]
        n_shots = pts[0][2] if pts else 0

        print(f"\n({L},{M})  N={N}  points={len(pts)}  shots/pt={n_shots:,}  "
              f"running {args.bootstrap} bootstrap iterations...", flush=True)

        p_star, mean_b, std_b, ci_lo, ci_hi = bootstrap_threshold_ci(
            pts, args.target_wer, args.bootstrap, rng
        )
        half = (ci_hi - ci_lo) / 2 if not np.isnan(ci_lo) else float("nan")
        print(f"  p* = {p_star:.5f}   95% CI [{ci_lo:.5f}, {ci_hi:.5f}]   "
              f"half-width = ±{half:.5f}   bootstrap std = {std_b:.5f}")

        # Also report binomial SE on each WER point near threshold
        pts_near = [
            (p, wilson_ci(f, s))
            for p, f, s in pts
            if abs(f / s - args.target_wer) < 0.15
        ]

        rows_json.append(
            {
                "L": L, "M": M, "N": N,
                "p_star": p_star,
                "ci_lo_95": ci_lo,
                "ci_hi_95": ci_hi,
                "ci_half_width": half,
                "bootstrap_std": std_b,
                "bootstrap_mean": mean_b,
                "n_bootstrap": args.bootstrap,
                "shots_per_point": n_shots,
                "n_points": len(pts),
                "target_wer": args.target_wer,
                "wer_points_near_threshold": [
                    {"p": p, "wer_ci_lo": lo, "wer_ci_hi": hi}
                    for p, (lo, hi) in pts_near
                ],
            }
        )
        table_lines.append(
            f"({L},{M}){'':<8} {N:>6}  {'?':>7} "
            f"{p_star:>8.5f} {ci_lo:>8.5f} {ci_hi:>8.5f} {half:>8.5f} "
            f"{n_shots:>10,} {std_b:>10.5f}"
        )

    table_lines.append("-" * 80)
    table_lines.append(
        f"Note: 95% bootstrap CI via {args.bootstrap} parametric (binomial) resampling iterations."
    )
    table_lines.append(
        "      Interpolation method: piecewise linear between adjacent p-grid points."
    )
    table_str = "\n".join(table_lines)
    print("\n" + table_str)

    # Shot-count sensitivity
    sensitivity_results = {}
    if args.shot_sensitivity:
        shot_counts = [5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000]
        print("\nRunning shot-count sensitivity analysis...")
        for key in sorted(data):
            L, M = key
            pts = data[key]
            if not pts:
                continue
            actual = pts[0][2]
            valid_counts = [s for s in shot_counts if s <= actual]
            sens = shot_sensitivity(pts, args.target_wer, valid_counts, args.bootstrap, rng)
            sensitivity_results[f"L{L}M{M}"] = {
                "actual_shots": actual,
                "ci_half_by_shots": sens,
            }
            print(f"  ({L},{M})  " + "  ".join(
                f"{s//1000}k→±{v:.4f}" if not np.isnan(v) else f"{s//1000}k→N/A"
                for s, v in sens.items()
            ))

    # Save outputs
    output = {
        "threshold_ci": rows_json,
        "target_wer": args.target_wer,
        "bootstrap_iterations": args.bootstrap,
        "seed": args.seed,
        "shot_sensitivity": sensitivity_results,
    }
    json_path = out_dir / "threshold_ci_table.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True, default=str)

    txt_path = out_dir / "threshold_ci_table.txt"
    txt_path.write_text(table_str, encoding="utf-8")

    # Shot sensitivity plot
    if HAS_MPL and args.shot_sensitivity and sensitivity_results:
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(sensitivity_results)))
        for (label, sens_data), color in zip(sorted(sensitivity_results.items()), colors):
            shots_list = sorted(sens_data["ci_half_by_shots"].keys())
            half_widths = [sens_data["ci_half_by_shots"][s] for s in shots_list]
            valid = [(s, h) for s, h in zip(shots_list, half_widths) if not np.isnan(h)]
            if valid:
                xs, ys = zip(*valid)
                ax.loglog(xs, ys, "o-", color=color, label=label, markersize=4)
        ax.axvline(200_000, color="grey", linestyle="--", linewidth=0.8, label="200k (current)")
        ax.set_xlabel("Shots per p-point", fontsize=12)
        ax.set_ylabel("CI half-width on p*", fontsize=12)
        ax.set_title("Shot-count sensitivity: CI width on pseudo-threshold", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        plot_path = out_dir / "shot_sensitivity.pdf"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"\nSaved: {plot_path}")

    print(f"\nSaved: {json_path}")
    print(f"Saved: {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
