#!/usr/bin/env python3
"""
Finite-Size Scaling (FSS) analysis for BB code erasure-channel thresholds.

This script addresses Reviewer Issue #1 (misuse of "threshold") by:
  1. Loading WER(p, N) data from CSV files produced by sweep_scaling.py
  2. Fitting all curves simultaneously to the FSS ansatz near the critical point:
         WER(p, N) ≈ f( (p - p*) · N^(1/ν) )
     where f is approximated by a low-degree polynomial (or a two-parameter
     logistic), and (p*, ν) are the free fit parameters.
  3. Extracting the asymptotic pseudo-threshold p* with a 95% bootstrap CI.
  4. Plotting the raw WER curves and the FSS collapse plot.
  5. Reporting: "These are finite-size WER crossing points that converge to
     an asymptotic value p* ≈ ... as N → ∞."

Usage:
  python examples/finite_size_scaling_analysis.py \\
      --csv results/scaling_*.csv \\
      --p-window 0.08          \\  # half-width around each pseudo-threshold
      --poly-degree 3          \\  # polynomial degree for f(x)
      --bootstrap 2000         \\  # bootstrap iterations for CI
      --out-dir results/fss

Outputs:
  results/fss/fss_results.json          -- fit parameters, CIs, per-size p*
  results/fss/plot_wer_curves.pdf       -- raw WER vs p
  results/fss/plot_fss_collapse.pdf     -- FSS collapse (all curves on one plot)
  results/fss/plot_pstar_vs_N.pdf       -- pseudo-threshold convergence
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.optimize as sco
import scipy.stats as scs

# Optional matplotlib import – skip plotting gracefully if not available
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(path: str) -> List[Dict]:
    """Load a CSV produced by sweep_scaling.py and return list of row dicts."""
    rows = []
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def collect_data(csv_paths: List[str]) -> Dict[Tuple[int, int], List[Tuple[float, int, int]]]:
    """
    Returns: { (L, M): [(p, fails, shots), ...] }  sorted by p
    Only includes rows with family == "bivariate_bicycle".
    """
    data: Dict[Tuple[int, int], List] = {}
    for path in csv_paths:
        for row in load_csv(path):
            if row.get("family", "") != "bivariate_bicycle":
                continue
            try:
                L = int(row["L"])
                M = int(row["M"])
                p = float(row["p"])
                fails = int(row["fails"])
                shots = int(row["shots"])
            except (KeyError, ValueError):
                continue
            key = (L, M)
            data.setdefault(key, []).append((p, fails, shots))
    for key in data:
        data[key].sort(key=lambda t: t[0])
    return data


# ---------------------------------------------------------------------------
# Wilson confidence interval for binomial proportion
# ---------------------------------------------------------------------------

def wilson_ci(fails: int, shots: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score CI for WER."""
    p_hat = fails / shots
    denom = 1 + z**2 / shots
    center = (p_hat + z**2 / (2 * shots)) / denom
    half = z * np.sqrt(p_hat * (1 - p_hat) / shots + z**2 / (4 * shots**2)) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


# ---------------------------------------------------------------------------
# Per-size pseudo-threshold from linear interpolation + Wilson CI propagation
# ---------------------------------------------------------------------------

def pseudo_threshold_from_data(
    pts: List[Tuple[float, int, int]],
    target_wer: float = 0.10,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Returns (p_star, p_lo, p_hi) from linear interpolation of WER vs p.
    p_lo / p_hi are propagated 95% Wilson CI bounds.
    """
    sorted_pts = sorted(pts, key=lambda t: t[0])
    for i in range(len(sorted_pts) - 1):
        p0, f0, s0 = sorted_pts[i]
        p1, f1, s1 = sorted_pts[i + 1]
        w0, w1 = f0 / s0, f1 / s1
        if (w0 - target_wer) * (w1 - target_wer) <= 0 and w0 != w1:
            t = (target_wer - w0) / (w1 - w0)
            p_star = p0 + t * (p1 - p0)
            # Propagate CI bounds on w0 and w1 through interpolation
            lo0, hi0 = wilson_ci(f0, s0)
            lo1, hi1 = wilson_ci(f1, s1)
            t_lo = (target_wer - hi0) / (lo1 - hi0) if (lo1 - hi0) != 0 else t
            t_hi = (target_wer - lo0) / (hi1 - lo0) if (hi1 - lo0) != 0 else t
            t_lo, t_hi = sorted([t_lo, t_hi])
            p_lo = p0 + max(0.0, min(1.0, t_lo)) * (p1 - p0)
            p_hi = p0 + max(0.0, min(1.0, t_hi)) * (p1 - p0)
            return p_star, p_lo, p_hi
    return None, None, None


# ---------------------------------------------------------------------------
# FSS fit
# ---------------------------------------------------------------------------

def fss_polynomial(x: np.ndarray, *coeffs) -> np.ndarray:
    """Evaluate polynomial f(x) = sum_i coeffs[i] * x^i."""
    result = np.zeros_like(x, dtype=float)
    for i, c in enumerate(coeffs):
        result += c * x**i
    return result


def fss_objective(
    params: np.ndarray,
    Ns: List[int],
    p_arrays: List[np.ndarray],
    wer_arrays: List[np.ndarray],
    poly_degree: int,
) -> float:
    """
    Residual sum-of-squares for the FSS ansatz:
        WER(p, N) ≈ f( (p - p*) · N^(1/ν) )
    params = [p_star, nu, c0, c1, ..., c_{poly_degree}]
    """
    p_star = params[0]
    nu = params[1]
    coeffs = params[2:]
    if nu <= 0:
        return 1e12
    total_sq = 0.0
    for N, ps, ws in zip(Ns, p_arrays, wer_arrays):
        xs = (ps - p_star) * N ** (1.0 / nu)
        predicted = fss_polynomial(xs, *coeffs)
        total_sq += np.sum((ws - predicted) ** 2)
    return total_sq


def fit_fss(
    data: Dict[Tuple[int, int], List[Tuple[float, int, int]]],
    target_wer: float = 0.10,
    p_window: float = 0.08,
    poly_degree: int = 3,
    n_restarts: int = 8,
) -> Dict:
    """
    Perform FSS fit over all BB code sizes.
    Restricts data to |p - p*(N)| < p_window to stay in the scaling regime.
    Returns dict with fit parameters, their uncertainties, and diagnostics.
    """
    # Step 1: get per-size pseudo-thresholds (used to center the window)
    per_size = {}
    for key in sorted(data):
        L, M = key
        N = 2 * L * M
        pts = data[key]
        p_star, p_lo, p_hi = pseudo_threshold_from_data(pts, target_wer)
        if p_star is None:
            print(f"  [FSS] Skipping ({L},{M}): no crossing found near WER={target_wer}")
            continue
        per_size[key] = {"N": N, "p_star": p_star, "p_lo": p_lo, "p_hi": p_hi}

    if len(per_size) < 3:
        print("[FSS] Need at least 3 code sizes for FSS fit.")
        return {"error": "insufficient_sizes"}

    # Rough initial guess: average of per-size p_stars
    p_star_init = float(np.mean([v["p_star"] for v in per_size.values()]))

    # Step 2: build windowed arrays
    Ns, p_arrs, wer_arrs, wer_errs = [], [], [], []
    for key in sorted(per_size):
        L, M = key
        N = per_size[key]["N"]
        ps = per_size[key]["p_star"]
        pts_full = data[key]
        # Select points within window of this size's p_star
        window = [(p, f, s) for p, f, s in pts_full if abs(p - ps) <= p_window]
        if len(window) < 3:
            # Use all points if window is too small
            window = pts_full
        p_vals = np.array([t[0] for t in window], dtype=float)
        wer_vals = np.array([t[1] / t[2] for t in window], dtype=float)
        err_vals = np.array([np.sqrt(t[1] * (t[2] - t[1]) / t[2] ** 3) for t in window], dtype=float)
        Ns.append(N)
        p_arrs.append(p_vals)
        wer_arrs.append(wer_vals)
        wer_errs.append(err_vals)

    # Step 3: optimise with multiple restarts
    n_params = 2 + poly_degree + 1  # p_star, nu, c0..c_d
    best_res = None
    best_cost = np.inf
    rng = np.random.default_rng(42)
    for _ in range(n_restarts):
        p0 = p_star_init + rng.uniform(-0.02, 0.02)
        nu0 = rng.uniform(0.5, 2.5)
        c0 = [target_wer] + [rng.uniform(-1, 1) for _ in range(poly_degree)]
        x0 = np.array([p0, nu0] + c0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sco.minimize(
                fss_objective,
                x0,
                args=(Ns, p_arrs, wer_arrs, poly_degree),
                method="Nelder-Mead",
                options={"maxiter": 50000, "xatol": 1e-7, "fatol": 1e-9},
            )
        if res.fun < best_cost:
            best_cost = res.fun
            best_res = res

    if best_res is None:
        return {"error": "optimisation_failed"}

    p_star_fit = float(best_res.x[0])
    nu_fit = float(best_res.x[1])
    coeffs_fit = list(best_res.x[2:])

    # Step 4: bootstrap uncertainty on p_star and nu
    print(f"[FSS] Running bootstrap ({_bootstrap_n} iterations)...")

    def _one_boot(rng_b):
        boot_data = {}
        for key, pts in data.items():
            # Resample shots using binomial
            boot_pts = []
            for p_val, f, s in pts:
                p_hat = f / s
                f_b = int(rng_b.binomial(s, p_hat))
                boot_pts.append((p_val, f_b, s))
            boot_data[key] = boot_pts
        # Re-compute per-size windows
        Ns_b, p_arrs_b, wer_arrs_b = [], [], []
        for key in sorted(per_size):
            L, M = key
            N = per_size[key]["N"]
            ps_b = pseudo_threshold_from_data(boot_data[key], target_wer)[0]
            if ps_b is None:
                ps_b = per_size[key]["p_star"]
            pts_w = [(p, f, s) for p, f, s in boot_data[key] if abs(p - ps_b) <= p_window]
            if len(pts_w) < 3:
                pts_w = boot_data[key]
            Ns_b.append(N)
            p_arrs_b.append(np.array([t[0] for t in pts_w], dtype=float))
            wer_arrs_b.append(np.array([t[1] / t[2] for t in pts_w], dtype=float))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_b = sco.minimize(
                fss_objective,
                best_res.x,
                args=(Ns_b, p_arrs_b, wer_arrs_b, poly_degree),
                method="Nelder-Mead",
                options={"maxiter": 20000, "xatol": 1e-6, "fatol": 1e-8},
            )
        return res_b.x[0], res_b.x[1]  # p_star, nu

    boot_p_stars, boot_nus = [], []
    rng_boot = np.random.default_rng(99)
    for i in range(_bootstrap_n):
        try:
            ps_b, nu_b = _one_boot(rng_boot)
            boot_p_stars.append(ps_b)
            boot_nus.append(nu_b)
        except Exception:
            pass

    boot_p_stars = np.array(boot_p_stars)
    boot_nus = np.array(boot_nus)
    p_star_ci = (float(np.percentile(boot_p_stars, 2.5)), float(np.percentile(boot_p_stars, 97.5)))
    nu_ci = (float(np.percentile(boot_nus, 2.5)), float(np.percentile(boot_nus, 97.5)))

    return {
        "p_star_fit": p_star_fit,
        "nu_fit": nu_fit,
        "p_star_ci_95": p_star_ci,
        "nu_ci_95": nu_ci,
        "poly_coeffs": coeffs_fit,
        "poly_degree": poly_degree,
        "p_window": p_window,
        "residual_ss": float(best_cost),
        "n_sizes": len(Ns),
        "per_size": {
            f"L{L}M{M}": {
                "N": v["N"],
                "p_star_interp": v["p_star"],
                "p_star_ci_lo": v["p_lo"],
                "p_star_ci_hi": v["p_hi"],
            }
            for (L, M), v in per_size.items()
        },
        "Ns_used": Ns,
    }


_bootstrap_n = 500  # default; overridden by --bootstrap arg


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_wer_curves(
    data: Dict[Tuple[int, int], List[Tuple[float, int, int]]],
    fss_result: Dict,
    out_path: Path,
    target_wer: float = 0.10,
) -> None:
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(data)))
    for (key, color) in zip(sorted(data), colors):
        L, M = key
        N = 2 * L * M
        pts = data[key]
        ps = np.array([t[0] for t in pts])
        ws = np.array([t[1] / t[2] for t in pts])
        lo = np.array([wilson_ci(t[1], t[2])[0] for t in pts])
        hi = np.array([wilson_ci(t[1], t[2])[1] for t in pts])
        ax.errorbar(
            ps, ws, yerr=[ws - lo, hi - ws],
            fmt="o-", color=color, capsize=3, markersize=4, linewidth=1.2,
            label=f"[[{2*L*M},{2*L*M//12 if M == L//2 else '?'},{L//2}]] N={N}",
        )
    ax.axhline(target_wer, color="grey", linestyle="--", linewidth=0.8, label=f"WER = {target_wer}")
    if "p_star_fit" in fss_result:
        ax.axvline(fss_result["p_star_fit"], color="red", linestyle=":",
                   linewidth=1.2, label=f"p* ≈ {fss_result['p_star_fit']:.4f} (FSS)")
    ax.set_xlabel("Erasure rate p", fontsize=12)
    ax.set_ylabel("Word error rate (WER)", fontsize=12)
    ax.set_title("Bivariate Bicycle Codes – Erasure Channel (BP-OSD)", fontsize=12)
    ax.legend(fontsize=8, framealpha=0.7)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] WER curves → {out_path}")


def plot_fss_collapse(
    data: Dict[Tuple[int, int], List[Tuple[float, int, int]]],
    fss_result: Dict,
    out_path: Path,
    p_window: float = 0.08,
) -> None:
    if not HAS_MPL or "p_star_fit" not in fss_result:
        return
    p_star = fss_result["p_star_fit"]
    nu = fss_result["nu_fit"]
    coeffs = fss_result["poly_coeffs"]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(data)))
    x_all = []
    for (key, color) in zip(sorted(data), colors):
        L, M = key
        N = 2 * L * M
        pts = sorted(data[key])
        xs, ws = [], []
        for p, fails, shots in pts:
            if abs(p - p_star) <= p_window * 1.2:
                xs.append((p - p_star) * N ** (1.0 / nu))
                ws.append(fails / shots)
                x_all.append(xs[-1])
        if xs:
            ax.scatter(xs, ws, color=color, s=20, label=f"N={N}", zorder=3)
    # Draw FSS curve
    if x_all:
        x_fit = np.linspace(min(x_all), max(x_all), 300)
        y_fit = fss_polynomial(x_fit, *coeffs)
        ax.plot(x_fit, y_fit, "r-", linewidth=1.5, label="FSS polynomial fit", zorder=4)
    ax.set_xlabel(r"$(p - p^*_\infty)\, N^{1/\nu}$", fontsize=12)
    ax.set_ylabel("Word error rate (WER)", fontsize=12)
    ax.set_title(
        f"FSS collapse:  $p^*_\\infty \\approx {p_star:.4f}$,  $\\nu \\approx {nu:.2f}$",
        fontsize=12,
    )
    ax.legend(fontsize=8, framealpha=0.7)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] FSS collapse → {out_path}")


def plot_pstar_vs_N(
    fss_result: Dict,
    p_star_fit: float,
    p_star_ci: Tuple[float, float],
    out_path: Path,
) -> None:
    if not HAS_MPL or "per_size" not in fss_result:
        return
    per_size = fss_result["per_size"]
    Ns = sorted([v["N"] for v in per_size.values()])
    ps_list, lo_list, hi_list = [], [], []
    for v in sorted(per_size.values(), key=lambda x: x["N"]):
        ps_list.append(v["p_star_interp"])
        lo_list.append(v.get("p_star_ci_lo") or v["p_star_interp"])
        hi_list.append(v.get("p_star_ci_hi") or v["p_star_interp"])

    fig, ax = plt.subplots(figsize=(6, 4))
    ps_arr = np.array(ps_list)
    lo_arr = np.minimum(np.array(lo_list), ps_arr)
    hi_arr = np.maximum(np.array(hi_list), ps_arr)
    yerr_lo = np.clip(ps_arr - lo_arr, 0, None)
    yerr_hi = np.clip(hi_arr - ps_arr, 0, None)
    ax.errorbar(Ns, ps_list, yerr=[yerr_lo, yerr_hi], fmt="o-", capsize=4,
                color="steelblue", label="Per-size WER crossing", zorder=3)
    ax.axhline(p_star_fit, color="red", linestyle="--",
               label=f"FSS asymptote $p^*_\\infty \\approx {p_star_fit:.4f}$")
    ax.fill_between(
        [min(Ns) * 0.9, max(Ns) * 1.1],
        [p_star_ci[0]] * 2, [p_star_ci[1]] * 2,
        color="red", alpha=0.12, label="95% CI (bootstrap)",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Physical qubits N", fontsize=12)
    ax.set_ylabel("Pseudo-threshold $p^*$", fontsize=12)
    ax.set_title("Finite-size convergence of WER crossing point", fontsize=12)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] p* vs N → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    global _bootstrap_n

    parser = argparse.ArgumentParser(
        description="Finite-size scaling analysis for BB code erasure thresholds."
    )
    parser.add_argument(
        "--csv", nargs="+", required=True,
        help="One or more CSV files from sweep_scaling.py (glob patterns ok if quoted).",
    )
    parser.add_argument("--target-wer", type=float, default=0.10)
    parser.add_argument(
        "--p-window", type=float, default=0.08,
        help="Half-width of p-window around each p* used for FSS fit (default: 0.08)",
    )
    parser.add_argument(
        "--poly-degree", type=int, default=3,
        help="Degree of polynomial f(x) in FSS ansatz (default: 3)",
    )
    parser.add_argument(
        "--bootstrap", type=int, default=500,
        help="Number of bootstrap iterations for CI (default: 500; use 2000+ for publication)",
    )
    parser.add_argument(
        "--out-dir", type=str, default="results/fss",
        help="Output directory for plots and JSON (default: results/fss)",
    )
    args = parser.parse_args()

    _bootstrap_n = args.bootstrap

    # Expand any glob patterns in --csv
    csv_paths: List[str] = []
    for pattern in args.csv:
        expanded = glob.glob(pattern)
        if expanded:
            csv_paths.extend(expanded)
        else:
            csv_paths.append(pattern)

    print(f"Loading data from {len(csv_paths)} CSV file(s)...")
    data = collect_data(csv_paths)
    print(f"Found {len(data)} code size(s): {sorted(data.keys())}")
    for key, pts in sorted(data.items()):
        L, M = key
        print(f"  ({L},{M})  N={2*L*M}  points={len(pts)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-size pseudo-thresholds
    print("\nPer-size WER crossing points:")
    per_size_results = {}
    for key in sorted(data):
        L, M = key
        N = 2 * L * M
        p_star, p_lo, p_hi = pseudo_threshold_from_data(data[key], args.target_wer)
        p_str   = f"{p_star:.5f}" if p_star is not None else "not found"
        p_lo_s  = f"{p_lo:.5f}"  if p_lo   is not None else "N/A"
        p_hi_s  = f"{p_hi:.5f}"  if p_hi   is not None else "N/A"
        print(f"  ({L},{M})  N={N}  p* = {p_str}  95% CI [{p_lo_s}, {p_hi_s}]")
        per_size_results[f"L{L}M{M}"] = {
            "L": L, "M": M, "N": N,
            "p_star": p_star, "ci_lo": p_lo, "ci_hi": p_hi,
        }

    # FSS fit
    print(f"\nRunning FSS fit (poly_degree={args.poly_degree}, p_window={args.p_window})...")
    fss_result = fit_fss(
        data,
        target_wer=args.target_wer,
        p_window=args.p_window,
        poly_degree=args.poly_degree,
    )

    if "error" in fss_result:
        print(f"[FSS] Error: {fss_result['error']}")
    else:
        p_star_fit = fss_result["p_star_fit"]
        nu_fit = fss_result["nu_fit"]
        ci = fss_result["p_star_ci_95"]
        nu_ci = fss_result["nu_ci_95"]
        print(f"\n=== FSS Results ===")
        print(f"  Asymptotic pseudo-threshold:  p*_inf = {p_star_fit:.5f}")
        print(f"  95% bootstrap CI:              [{ci[0]:.5f}, {ci[1]:.5f}]")
        print(f"  Critical exponent:             nu     = {nu_fit:.3f}")
        print(f"  95% bootstrap CI:              [{nu_ci[0]:.3f}, {nu_ci[1]:.3f}]")
        print(f"  Residual SS:                   {fss_result['residual_ss']:.6e}")

    # Save JSON
    output = {
        "per_size_pseudo_thresholds": per_size_results,
        "fss_fit": fss_result,
        "target_wer": args.target_wer,
        "poly_degree": args.poly_degree,
        "p_window": args.p_window,
        "bootstrap_iterations": args.bootstrap,
        "csv_files": csv_paths,
    }
    json_path = out_dir / "fss_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True, default=str)
    print(f"\nSaved: {json_path}")

    # Plots
    if HAS_MPL:
        plot_wer_curves(
            data, fss_result,
            out_dir / "plot_wer_curves.pdf",
            target_wer=args.target_wer,
        )
        plot_fss_collapse(
            data, fss_result,
            out_dir / "plot_fss_collapse.pdf",
            p_window=args.p_window,
        )
        if "p_star_fit" in fss_result:
            plot_pstar_vs_N(
                fss_result,
                fss_result["p_star_fit"],
                fss_result["p_star_ci_95"],
                out_dir / "plot_pstar_vs_N.pdf",
            )
    else:
        print("[warn] matplotlib not found; skipping plots.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
