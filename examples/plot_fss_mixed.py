#!/usr/bin/env python3
"""
Standalone plot generator for mixed-channel FSS results.
Reads results/fss_mixed/fss_mixed_results.json and produces:
  - plot_pstar_degradation.png   (key figure: p*_inf vs delta/eps)
  - plot_pstar_vs_N_deXXX.png   (per delta/eps: p*(N) convergence)

Usage:
  python examples/plot_fss_mixed.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
JSON_PATH  = ROOT / "results" / "fss_mixed" / "fss_mixed_results.json"
OUT_DIR    = ROOT / "results" / "fss_mixed"

# ── colour palette ─────────────────────────────────────────────────────────
MAROON  = "#8b1a1a"
BLUE    = "steelblue"
GREY    = "#888888"
CRIMSON = "crimson"


def load_results(path: Path) -> dict:
    with open(path) as f:
        raw = json.load(f)
    # keys are strings like "0.1", "0.3", …
    return {float(k): v["fss"] for k, v in raw.items()}


# ---------------------------------------------------------------------------
# 1.  p*(N) convergence  —  one panel per delta/eps
# ---------------------------------------------------------------------------

def plot_pstar_vs_N_one(fss: dict, de: float, out_path: Path) -> None:
    """p*(N) convergence + FSS fit for one delta/eps value."""
    entries = sorted(fss["per_size"].values(), key=lambda v: v["N"])
    Ns  = np.array([v["N"]              for v in entries])
    ps  = np.array([v["p_star_interp"]  for v in entries])
    lo  = np.array([v.get("p_lo", v["p_star_interp"]) for v in entries])
    hi  = np.array([v.get("p_hi", v["p_star_interp"]) for v in entries])

    p_inf = fss["p_star_fit"]
    a_fit = fss["a_fit"]
    nu    = fss["nu_fit"]
    ci    = fss["p_star_ci_95"]

    fig, ax = plt.subplots(figsize=(5, 4))

    ax.errorbar(Ns, ps,
                yerr=[np.clip(ps - lo, 0, None), np.clip(hi - ps, 0, None)],
                fmt="o", capsize=4, color=BLUE, zorder=3,
                label=r"$p^*(N)$ (WER = 0.10)")

    # FSS curve
    Nfine = np.linspace(Ns.min(), Ns.max() * 3, 400)
    ax.plot(Nfine, p_inf - a_fit * Nfine ** (-1.0 / nu),
            "r--", lw=1.5,
            label=(r"$p^*_\infty = " + f"{p_inf:.3f}$, "
                   r"$\nu = " + f"{nu:.2f}$"))
    ax.axhline(p_inf, color=CRIMSON, lw=0.8, alpha=0.4, ls=":")
    ax.fill_between(
        [Ns.min() * 0.8, Ns.max() * 3],
        [ci[0]] * 2, [ci[1]] * 2,
        color=CRIMSON, alpha=0.10, label="95% CI (bootstrap)")

    ax.set_xscale("log")
    ax.set_xlabel("$N$ (physical qubits)", fontsize=11)
    ax.set_ylabel("$p^*$ (WER = 0.10 crossing)", fontsize=11)
    ax.set_title(
        r"$\delta/\varepsilon = " + f"{de:.2f}$\n"
        r"$p^*_\infty = " + f"{p_inf:.3f}$   "
        r"$\nu = " + f"{nu:.2f}$",
        fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ---------------------------------------------------------------------------
# 2.  Degradation curve  —  KEY FIGURE
# ---------------------------------------------------------------------------

def plot_degradation(results: dict, out_path: Path) -> None:
    """
    p*_inf vs delta/eps:
      - blue scatter dots = per-size finite-N crossings
      - red squares + errorbars = FSS-extrapolated asymptote
      - dotted line = pure-erasure v1 result (0.488)
    """
    # All code sizes for colour scale
    all_Ns = sorted({
        v["N"]
        for fss in results.values()
        for v in fss.get("per_size", {}).values()
    })
    cmap = plt.cm.Blues
    size_colors = {N: cmap(0.35 + 0.45 * i / max(len(all_Ns) - 1, 1))
                   for i, N in enumerate(all_Ns)}

    des = sorted(results.keys())

    fig, ax = plt.subplots(figsize=(7, 5))

    # Per-size finite-N crossings
    for de in des:
        fss = results[de]
        for v in fss.get("per_size", {}).values():
            N = v["N"]
            ax.scatter([de], [v["p_star_interp"]],
                       color=size_colors.get(N, GREY),
                       s=28, zorder=3, alpha=0.80)

    # Dummy legend entries for code sizes
    for N, c in size_colors.items():
        ax.scatter([], [], color=c, s=28, alpha=0.80,
                   label=f"$N = {N}$ (finite-size crossing)")

    # FSS asymptotes with CI errorbars
    des_ok = [de for de in des if "p_star_fit" in results[de]]
    p_inf  = [results[de]["p_star_fit"]      for de in des_ok]
    ci_lo  = [results[de]["p_star_ci_95"][0] for de in des_ok]
    ci_hi  = [results[de]["p_star_ci_95"][1] for de in des_ok]

    ax.errorbar(
        des_ok, p_inf,
        yerr=[np.array(p_inf) - np.array(ci_lo),
              np.array(ci_hi) - np.array(p_inf)],
        fmt="s-", color=CRIMSON, capsize=5, linewidth=2,
        markersize=8, zorder=5,
        label=r"$p^*_\infty$ (FSS, $N \to \infty$)")

    # Pure-erasure reference from v1
    ax.axhline(0.488, color=GREY, lw=1.0, ls="--",
               label=r"$\delta/\varepsilon = 0$ (pure erasure, $p^*_\infty = 0.488$, v1)")

    ax.set_xlabel(r"$\delta/\varepsilon$  (depolarizing fraction)", fontsize=12)
    ax.set_ylabel(r"Asymptotic threshold $p^*_\infty$  (WER = 0.10)", fontsize=12)
    ax.set_title(
        "Threshold degradation: mixed erasure + depolarizing noise\n"
        r"BB codes, BP-OSD-5;  FSS via $p^*(N)=p^*_\infty - a\,N^{-1/\nu}$ [Dennis et al. 2002]",
        fontsize=11)
    ax.legend(fontsize=8, framealpha=0.88, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not JSON_PATH.exists():
        print(f"JSON not found: {JSON_PATH}"); sys.exit(1)

    results = load_results(JSON_PATH)
    print(f"Loaded FSS results for delta/eps = {sorted(results.keys())}\n")

    # Per-size convergence plots
    for de, fss in sorted(results.items()):
        if "per_size" not in fss:
            continue
        tag = f"de{de:.3f}".replace(".", "p")
        out = OUT_DIR / f"plot_pstar_vs_N_{tag}.png"
        plot_pstar_vs_N_one(fss, de, out)

    # Key degradation figure
    plot_degradation(results, OUT_DIR / "plot_pstar_degradation.png")

    print("\nAll plots written to", OUT_DIR)


if __name__ == "__main__":
    main()
