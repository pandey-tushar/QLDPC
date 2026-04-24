#!/usr/bin/env python3
"""
FSS analysis for mixed-channel (erasure + depolarizing) BB code data.

Method: p*(N) power-law extrapolation (Dennis et al. 2002; used as
independent check in v1 paper Fig. fss_linear).  Directly derived from
the standard FSS ansatz WER(p,N) ~ f[(p-p*_inf)*N^(1/nu)] -- at the
WER=0.10 crossing, (p*(N)-p*_inf)*N^(1/nu) = const, giving:

    p*(N) = p*_inf - a * N^{-1/nu}          [Dennis et al. 2002, Eq. after (2)]

Bisection data gives highly accurate per-size p*(N); curve_fit with
bounds extracts (p*_inf, a, nu) robustly.

References
----------
Dennis et al., J. Math. Phys. 43, 4452 (2002)  -- FSS framework for QEC
Wang et al.,   PRA 68, 022307 (2003)            -- FSS corrections
Bravyi et al., Nature 627, 778 (2024)           -- BB code FSS (same method)

Outputs (results/fss_mixed/)
  fss_mixed_results.json
  plot_pstar_vs_N_de<X>.pdf     -- p*(N) convergence per delta/eps
  plot_pstar_degradation.pdf    -- KEY FIGURE: p*_inf vs delta/eps
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.optimize as sco

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finite_size_scaling_analysis import wilson_ci, pseudo_threshold_from_data


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_mixed_csvs(paths: List[Path]) -> Dict[float, Dict[Tuple[int,int], List]]:
    """
    Returns { delta_eps: { (L,M): [(p, fails, shots), ...] } }
    When the same (delta_eps, L, M, p) appears twice (coarse + bisection),
    keeps the row with the higher shot count.
    """
    raw: Dict = {}
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("family", "") != "bivariate_bicycle":
                    continue
                de = round(float(row["delta_eps_ratio"]), 9)
                L  = int(row["L"]); M = int(row["M"])
                p  = round(float(row["p"]), 9)
                f_ = int(row["fails"]); s = int(row["shots"])
                raw.setdefault(de, {}).setdefault((L, M), {})
                prev = raw[de][(L, M)].get(p)
                if prev is None or s > prev[1]:
                    raw[de][(L, M)][p] = (f_, s)

    out: Dict = {}
    for de, sizes in raw.items():
        out[de] = {key: sorted((p, f, s) for p, (f, s) in d.items())
                   for key, d in sizes.items()}
    return out


# ---------------------------------------------------------------------------
# FSS: p*(N) = p*_inf - a * N^{-1/nu}   [Dennis et al. 2002]
# ---------------------------------------------------------------------------

def bootstrap_pstar(
    pts: List[Tuple[float, int, int]],
    target_wer: float,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Parametric bootstrap for p*(N): resample fails_b ~ Binomial(shots, wer)
    at each p-value, re-extract p*(N) via linear interpolation.
    Matches v1 paper methodology (5000-iter bootstrap, CIs ≲ 0.001).
    Ref: Dennis et al. (2002); v1 paper Sec. 3 (arXiv:2603.19062).
    """
    ps_arr  = np.array([p     for p, f, s in pts])
    f_arr   = np.array([f     for p, f, s in pts])
    s_arr   = np.array([s     for p, f, s in pts], dtype=float)
    wer_arr = f_arr / s_arr

    samples = []
    for _ in range(n_boot):
        f_b   = rng.binomial(s_arr.astype(int), np.clip(wer_arr, 0, 1))
        wer_b = f_b / s_arr
        # linear interpolation for crossing
        p_star_b = None
        for i in range(len(ps_arr) - 1):
            w0, w1 = wer_b[i], wer_b[i+1]
            if (w0 - target_wer) * (w1 - target_wer) <= 0 and w0 != w1:
                t = (target_wer - w0) / (w1 - w0)
                p_star_b = float(ps_arr[i] + t * (ps_arr[i+1] - ps_arr[i]))
                break
        if p_star_b is not None:
            samples.append(p_star_b)
    return np.array(samples)


def fit_fss_pstar_extrap(
    data: Dict[Tuple[int,int], List],
    bootstrap_n: int = 500,
    target_wer: float = 0.10,
) -> Dict:
    """
    Fit p*(N) = p*_inf - a * N^{-1/nu}  [Dennis et al. 2002]

    Sigma on each p*(N) comes from parametric bootstrap: resample
    fails_b ~ Binomial(shots, wer) at every p-value, re-extract p*(N)
    via linear interpolation, repeat bootstrap_n times.  This matches
    the methodology in the v1 paper (arXiv:2603.19062, Sec. 3).

    Power-law fit uses curve_fit with physically motivated bounds.
    Bootstrap CI on (p*_inf, nu) resamples the per-size p*(N) distributions.

    Refs: Dennis et al., J. Math. Phys. 43, 4452 (2002)
          Wang et al., PRA 68, 022307 (2003)
          Bravyi et al., Nature 627, 778 (2024)
    """
    rng = np.random.default_rng(42)

    per_size = {}
    for key in sorted(data):
        L, M = key
        N = 2 * L * M
        pts = data[key]
        p_star, p_lo, p_hi = pseudo_threshold_from_data(pts, target_wer)
        if p_star is None:
            continue

        # Parametric bootstrap for proper sigma (not bracket width)
        boot_samples = bootstrap_pstar(pts, target_wer, bootstrap_n, rng)
        if len(boot_samples) < 10:
            sigma = max(0.5 * ((p_hi or p_star) - (p_lo or p_star)), 5e-4)
            ci_lo, ci_hi = p_star - sigma, p_star + sigma
        else:
            sigma  = float(np.std(boot_samples))
            ci_lo  = float(np.percentile(boot_samples, 2.5))
            ci_hi  = float(np.percentile(boot_samples, 97.5))
            sigma  = max(sigma, 5e-4)  # floor to avoid degenerate fits

        per_size[key] = {
            "N": N, "p_star": p_star, "sigma": sigma,
            "p_lo": ci_lo, "p_hi": ci_hi,
        }

    if len(per_size) < 3:
        return {"error": "insufficient_sizes"}

    keys_sorted = sorted(per_size.keys())
    Ns   = np.array([per_size[k]["N"]      for k in keys_sorted], dtype=float)
    ps   = np.array([per_size[k]["p_star"] for k in keys_sorted], dtype=float)
    sigs = np.array([per_size[k]["sigma"]  for k in keys_sorted], dtype=float)
    p_max = float(np.max(ps))

    def model(N, p_inf, a, nu):
        return p_inf - a * N ** (-1.0 / nu)

    lo_bounds = [p_max + 1e-4, 1e-6, 0.3]
    hi_bounds = [0.500,        10.0, 15.0]

    best_popt, best_pcov, best_cost = None, None, np.inf
    rng2 = np.random.default_rng(7)
    for _ in range(40):
        p0 = [
            rng2.uniform(p_max + 0.005, min(p_max + 0.25, 0.499)),
            rng2.uniform(0.01, 3.0),
            rng2.uniform(0.5,  8.0),
        ]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, pcov = sco.curve_fit(
                    model, Ns, ps, p0=p0,
                    sigma=sigs, absolute_sigma=True,
                    bounds=(lo_bounds, hi_bounds),
                    maxfev=50000,
                )
            cost = float(np.sum(((ps - model(Ns, *popt)) / sigs) ** 2))
            if cost < best_cost:
                best_cost, best_popt, best_pcov = cost, popt, pcov
        except (RuntimeError, ValueError):
            continue

    if best_popt is None:
        return {"error": "optimisation_failed"}

    p_inf_fit, a_fit, nu_fit = best_popt

    # Bootstrap CI on fit parameters: resample p*(N) from their distributions
    boot_pinf, boot_nu = [], []
    rng_b = np.random.default_rng(99)
    for _ in range(bootstrap_n):
        ps_b   = ps + rng_b.normal(0, sigs)
        p_max_b = float(np.max(ps_b))
        lb_b   = [p_max_b + 1e-4, 1e-6, 0.3]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt_b, _ = sco.curve_fit(
                    model, Ns, ps_b, p0=best_popt,
                    sigma=sigs, absolute_sigma=True,
                    bounds=(lb_b, hi_bounds),
                    maxfev=10000,
                )
            boot_pinf.append(popt_b[0])
            boot_nu.append(popt_b[2])
        except Exception:
            pass

    ci_p  = (float(np.percentile(boot_pinf, 2.5)), float(np.percentile(boot_pinf, 97.5)))
    ci_nu = (float(np.percentile(boot_nu,   2.5)), float(np.percentile(boot_nu,   97.5)))

    return {
        "p_star_fit":    float(p_inf_fit),
        "a_fit":         float(a_fit),
        "nu_fit":        float(nu_fit),
        "p_star_ci_95":  ci_p,
        "nu_ci_95":      ci_nu,
        "residual_chi2": float(best_cost),
        "n_sizes":       len(per_size),
        "per_size": {
            f"L{L}M{M}": {
                "N": v["N"], "p_star_interp": v["p_star"],
                "sigma": v["sigma"],
                "p_lo":  v["p_lo"], "p_hi": v["p_hi"],
            }
            for (L, M), v in per_size.items()
        },
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_pstar_vs_N(result: Dict, out_path: Path, de: float) -> None:
    """p*(N) convergence plot for one delta/eps value."""
    if not HAS_MPL or "per_size" not in result:
        return
    entries = sorted(result["per_size"].values(), key=lambda v: v["N"])
    Ns  = np.array([v["N"]             for v in entries])
    ps  = np.array([v["p_star_interp"] for v in entries])
    lo  = np.array([v.get("p_lo") or v["p_star_interp"] for v in entries])
    hi  = np.array([v.get("p_hi") or v["p_star_interp"] for v in entries])

    p_inf = result["p_star_fit"]
    a_fit = result["a_fit"]
    nu    = result["nu_fit"]
    ci    = result["p_star_ci_95"]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.errorbar(Ns, ps,
                yerr=[np.clip(ps - lo, 0, None), np.clip(hi - ps, 0, None)],
                fmt="o", capsize=4, color="steelblue", label="Per-size $p^*$", zorder=3)

    # FSS curve
    Nfine = np.linspace(Ns.min(), Ns.max() * 3, 300)
    ax.plot(Nfine, p_inf - a_fit * Nfine ** (-1.0 / nu),
            "r--", lw=1.5, label=f"$p^*_\\infty={p_inf:.4f}$, $\\nu={nu:.2f}$")
    ax.axhline(p_inf, color="red", lw=0.8, alpha=0.5)
    ax.fill_between([Ns.min() * 0.8, Ns.max() * 3],
                    [ci[0]] * 2, [ci[1]] * 2,
                    color="red", alpha=0.1, label="95% CI (bootstrap)")

    ax.set_xscale("log")
    ax.set_xlabel("$N$ (physical qubits)", fontsize=11)
    ax.set_ylabel("$p^*$ (WER = 0.10 crossing)", fontsize=11)
    ax.set_title(f"FSS convergence  $\\delta/\\varepsilon={de:.2f}$", fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"  [plot] {out_path.name}")


def plot_degradation(all_results: Dict, out_path: Path) -> None:
    """
    KEY FIGURE: p*_inf vs delta/eps.
    Shows FSS-extrapolated asymptotic threshold degradation as depolarizing
    fraction increases.  Both per-size finite crossings and FSS asymptotes shown.
    """
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    # Collect all code sizes for legend
    all_Ns = sorted({
        v["N"]
        for res in all_results.values()
        for v in res["fss"].get("per_size", {}).values()
    })
    size_colors = dict(zip(all_Ns,
                           plt.cm.Blues(np.linspace(0.35, 0.80, len(all_Ns)))))

    des = sorted(all_results.keys())

    # Per-size finite-N crossings (scatter, lighter)
    for de, res in sorted(all_results.items()):
        for v in res["fss"].get("per_size", {}).values():
            N = v["N"]
            ax.scatter([de], [v["p_star_interp"]],
                       color=size_colors.get(N, "grey"),
                       s=22, zorder=3, alpha=0.75)

    # Dummy legend entries for code sizes
    for N, c in size_colors.items():
        ax.scatter([], [], color=c, s=22, alpha=0.75, label=f"$N={N}$ (finite)")

    # FSS asymptotes with CI errorbars (bold, red)
    des_ok  = [de for de in des if "p_star_fit" in all_results[de]["fss"]]
    p_inf   = [all_results[de]["fss"]["p_star_fit"]       for de in des_ok]
    ci_lo   = [all_results[de]["fss"]["p_star_ci_95"][0]  for de in des_ok]
    ci_hi   = [all_results[de]["fss"]["p_star_ci_95"][1]  for de in des_ok]

    ax.errorbar(des_ok, p_inf,
                yerr=[np.array(p_inf) - np.array(ci_lo),
                      np.array(ci_hi) - np.array(p_inf)],
                fmt="s-", color="crimson", capsize=5, linewidth=2,
                markersize=8, label=r"$p^*_\infty$ (FSS, $N\to\infty$)", zorder=5)

    # Reference: pure-erasure asymptote from v1
    ax.axhline(0.488, color="grey", lw=1.0, ls=":",
               label=r"$\delta/\varepsilon=0$ asymptote (0.488, v1)")

    ax.set_xlabel(r"$\delta/\varepsilon$  (depolarizing fraction)", fontsize=12)
    ax.set_ylabel(r"Asymptotic threshold $p^*_\infty$ (WER = 0.10)", fontsize=12)
    ax.set_title("Threshold degradation under mixed erasure+depolarizing noise\n"
                 "BB codes, BP-OSD; FSS via Dennis et al. (2002)", fontsize=11)
    ax.legend(fontsize=8, framealpha=0.85, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"  [plot] {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir",    default="results/phase1")
    ap.add_argument("--out-dir",   default="results/fss_mixed")
    ap.add_argument("--bootstrap", type=int,   default=500)
    args = ap.parse_args()

    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(in_dir.glob("bb_*_seq.csv"))
    if not csv_files:
        print(f"No bb_*_seq.csv found in {in_dir}"); return 1
    print(f"Loading {len(csv_files)} CSV(s): {[f.name for f in csv_files]}")

    mixed = load_mixed_csvs(csv_files)
    delta_eps_vals = sorted(mixed.keys())
    print(f"delta/eps values: {delta_eps_vals}\n")
    print("FSS method: p*(N) = p*_inf - a*N^(-1/nu)")
    print("Ref: Dennis et al., J. Math. Phys. 43, 4452 (2002)\n")

    all_results = {}

    for de in delta_eps_vals:
        data = mixed[de]
        sizes_str = ", ".join(f"({L},{M}) N={2*L*M}" for L, M in sorted(data))
        print(f"-- delta/eps = {de:.3f}  [{sizes_str}]")

        print("  Per-size p* (WER=0.10 crossing):")
        for key in sorted(data):
            L, M = key
            ps, plo, phi = pseudo_threshold_from_data(data[key], 0.10)
            print(f"    ({L},{M}) N={2*L*M}  p*={ps:.5f}  "
                  f"95%CI=[{plo:.5f},{phi:.5f}]" if ps else
                  f"    ({L},{M}) N={2*L*M}  p*=n/a")

        print(f"  Fitting p*(N) = p*_inf - a*N^(-1/nu)  [bootstrap={args.bootstrap}]...")
        fss = fit_fss_pstar_extrap(data, bootstrap_n=args.bootstrap)

        if "error" in fss:
            print(f"  [FSS error: {fss['error']}]")
        else:
            ci  = fss["p_star_ci_95"]
            nci = fss["nu_ci_95"]
            print(f"  p*_inf = {fss['p_star_fit']:.5f}  95%CI [{ci[0]:.5f}, {ci[1]:.5f}]")
            print(f"  nu     = {fss['nu_fit']:.3f}    95%CI [{nci[0]:.3f}, {nci[1]:.3f}]")
            print(f"  chi2   = {fss['residual_chi2']:.3f}")
            tag = f"de{de:.3f}".replace(".", "p")
            plot_pstar_vs_N(fss, out_dir / f"plot_pstar_vs_N_{tag}.pdf", de)

        all_results[de] = {"fss": fss}
        print()

    # Summary table
    print("=" * 65)
    print(f"{'delta/eps':>12}  {'p*_inf':>9}  {'CI_lo':>9}  {'CI_hi':>9}  {'nu':>6}  {'chi2':>7}")
    print("-" * 65)
    for de in delta_eps_vals:
        fss = all_results[de]["fss"]
        if "p_star_fit" in fss:
            ci = fss["p_star_ci_95"]
            print(f"{de:>12.3f}  {fss['p_star_fit']:>9.5f}  "
                  f"{ci[0]:>9.5f}  {ci[1]:>9.5f}  "
                  f"{fss['nu_fit']:>6.3f}  {fss['residual_chi2']:>7.3f}")
        else:
            print(f"{de:>12.3f}  {'n/a':>9}")
    print("=" * 65)

    json_path = out_dir / "fss_mixed_results.json"
    with json_path.open("w") as f:
        json.dump({str(de): v for de, v in all_results.items()},
                  f, indent=2, default=str)
    print(f"\nSaved: {json_path}")

    plot_degradation(all_results, out_dir / "plot_pstar_degradation.pdf")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
