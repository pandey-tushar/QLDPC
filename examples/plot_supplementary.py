#!/usr/bin/env python3
"""
Generate four supplementary publication figures:
  1. Linearized FSS:  p* vs N^{-1/nu}  (linear extrapolation to p*_inf)
  2. Rate-threshold trade-off:  K/N vs p*  (Pareto scatter)
  3. Overhead-threshold scatter:  N/K vs p*  (BB + surface codes)
  4. Fraction of capacity:  p* / (1 - K/N)  vs N  (how close to Shannon limit)
"""
from __future__ import annotations
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Hard-coded data (from results JSON / CI table)
# ---------------------------------------------------------------------------
BB = [
    # (L, M,  N,   K,   p_star,   ci_lo,     ci_hi)
    (12,  6,  144,  12,  0.37007,  0.36971,  0.37028),
    (18,  9,  324,   8,  0.43855,  0.43804,  0.43867),
    (24, 12,  576,  16,  0.44529,  0.44520,  0.44545),
    (30, 15,  900,   8,  0.46742,  0.46730,  0.46756),
    (36, 18, 1296,  12,  0.47060,  0.47051,  0.47070),
]

# Erasure-aware MWPM surface code (K=2 toric, 50k shots; L=36 confirmed 200k)
SC_AWARE = [
    # (L,  N,   K,  p_star)
    (12,  288,  2,  0.4093),
    (18,  648,  2,  0.4329),
    (24, 1152,  2,  0.4457),
    (30, 1800,  2,  0.4539),
    (36, 2592,  2,  0.4598),
]

FSS_P_INF  = 0.48784
FSS_NU     = 1.17912
FSS_CI_LO  = 0.48765
FSS_CI_HI  = 0.48800

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
PAPER_RC = {
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "lines.linewidth": 1.4,
    "lines.markersize": 6,
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linestyle": "--",
}

BB_COLOR  = "steelblue"
SC_COLOR  = "darkorange"
FIT_COLOR = "crimson"

# ---------------------------------------------------------------------------
# Figure 1 — Linearised FSS:  p* vs N^{-1/nu}
# ---------------------------------------------------------------------------
def plot_linearised_fss(out_path: Path) -> None:
    Ns      = np.array([r[2] for r in BB])
    pstars  = np.array([r[4] for r in BB])
    ci_lo   = np.array([r[5] for r in BB])
    ci_hi   = np.array([r[6] for r in BB])
    labels  = [f"${r[0]}\\times{r[1]}$\n$K={r[3]}$" for r in BB]

    inv_nu  = 1.0 / FSS_NU
    xs      = Ns ** (-inv_nu)                 # N^{-1/nu}

    # Linear fit: p* = p_inf + c * N^{-1/nu}
    coeffs  = np.polyfit(xs, pstars, 1)
    c_fit, p_inf_fit = coeffs

    x_ext   = np.linspace(0, xs.max() * 1.05, 300)
    y_fit   = np.polyval(coeffs, x_ext)

    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    # Shaded FSS CI band at x = 0
    ax.axhspan(FSS_CI_LO, FSS_CI_HI, xmin=0, xmax=0.04,
               color=FIT_COLOR, alpha=0.18, zorder=1)
    ax.axhline(FSS_P_INF, color=FIT_COLOR, linestyle="--", linewidth=1.1,
               label=f"$p^*_\\infty = {FSS_P_INF:.3f}$ (FSS fit)", zorder=2)

    # Linear extrapolation line
    ax.plot(x_ext, y_fit, color="dimgray", linestyle=":", linewidth=1.1,
            label=f"Linear fit ($c = {c_fit:.1f}$)", zorder=3)

    # Data points with CI bars
    yerr = [pstars - ci_lo, ci_hi - pstars]
    ax.errorbar(xs, pstars, yerr=yerr,
                fmt="o", color=BB_COLOR, capsize=4, capthick=1.2,
                zorder=5, label="BB pseudo-threshold $p^*$")

    for x, y, lab in zip(xs, pstars, labels):
        ax.annotate(lab, (x, y),
                    textcoords="offset points", xytext=(6, -14),
                    fontsize=7.5, color="black", ha="left")

    ax.set_xlabel(r"$N^{-1/\nu}$  ($\nu = 1.18$)", fontsize=11)
    ax.set_ylabel(r"Pseudo-threshold $p^*$", fontsize=11)
    ax.set_title("Linearised finite-size scaling of $p^*$", fontsize=11)
    ax.set_xlim(-0.0005, xs.max() * 1.12)
    ax.set_ylim(0.35, 0.505)
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"[plot] Linearised FSS -> {out_path}")
    print(f"       Linear-fit p*_inf = {p_inf_fit:.4f}  (FSS nonlinear = {FSS_P_INF:.4f})")


# ---------------------------------------------------------------------------
# Figure 2 — Rate vs pseudo-threshold  (K/N vs p*)
# ---------------------------------------------------------------------------
def plot_rate_vs_threshold(out_path: Path) -> None:
    Ns     = np.array([r[2] for r in BB])
    Ks     = np.array([r[3] for r in BB])
    ps     = np.array([r[4] for r in BB])
    labels = [f"${r[0]}\\times{r[1]}$" for r in BB]

    sc_ps  = np.array([r[3] for r in SC_AWARE])
    sc_Ks  = np.full(len(SC_AWARE), 2)
    sc_Ns  = np.array([r[1] for r in SC_AWARE])
    sc_lab = [f"$L={r[0]}$" for r in SC_AWARE]

    rates    = Ks  / Ns
    sc_rates = sc_Ks / sc_Ns

    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    ax.scatter(ps, rates, color=BB_COLOR, s=60, zorder=5, label="BB codes")
    for x, y, lab in zip(ps, rates, labels):
        ax.annotate(lab, (x, y), textcoords="offset points",
                    xytext=(5, 4), fontsize=8, color=BB_COLOR)

    ax.scatter(sc_ps, sc_rates, color=SC_COLOR, s=50, marker="s",
               zorder=5, label="Surface (erasure-aware MWPM)")
    for x, y, lab in zip(sc_ps, sc_rates, sc_lab):
        ax.annotate(lab, (x, y), textcoords="offset points",
                    xytext=(5, -12), fontsize=8, color=SC_COLOR)

    # Shannon hashing bound line: K/N = 1 - p  (rate <= 1-p for erasure channel)
    p_range = np.linspace(0.30, 0.52, 200)
    ax.plot(p_range, 1.0 - p_range, color="gray", linestyle="--",
            linewidth=1.1, label="Shannon limit $K/N = 1 - p$", zorder=2)

    ax.axvline(FSS_P_INF, color=FIT_COLOR, linestyle=":", linewidth=1.0,
               label=f"$p^*_\\infty = {FSS_P_INF:.3f}$")

    ax.set_xlabel(r"Pseudo-threshold $p^*$", fontsize=11)
    ax.set_ylabel(r"Encoding rate $K/N$", fontsize=11)
    ax.set_title("Rate–threshold trade-off", fontsize=11)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_xlim(0.36, 0.52)
    ax.set_ylim(-0.002, 0.092)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"[plot] Rate vs threshold -> {out_path}")


# ---------------------------------------------------------------------------
# Figure 3 — Overhead–threshold scatter (N/K vs p*)
# ---------------------------------------------------------------------------
def plot_overhead_threshold(out_path: Path) -> None:
    ps_bb  = np.array([r[4] for r in BB])
    Ns_bb  = np.array([r[2] for r in BB])
    Ks_bb  = np.array([r[3] for r in BB])
    ov_bb  = Ns_bb / Ks_bb
    lab_bb = [f"$[[{r[2]},{r[3]}]]$" for r in BB]

    ps_sc  = np.array([r[3] for r in SC_AWARE])
    Ns_sc  = np.array([r[1] for r in SC_AWARE])
    ov_sc  = Ns_sc / 2.0           # K=2 for all toric codes
    lab_sc = [f"$L={r[0]}$" for r in SC_AWARE]

    fig, ax = plt.subplots(figsize=(5.5, 4.2))

    ax.scatter(ps_bb, ov_bb, color=BB_COLOR, s=70, zorder=5,
               label="BB codes")
    for x, y, lab in zip(ps_bb, ov_bb, lab_bb):
        ax.annotate(lab, (x, y), textcoords="offset points",
                    xytext=(5, 4), fontsize=7.5, color=BB_COLOR)

    ax.scatter(ps_sc, ov_sc, color=SC_COLOR, s=55, marker="s", zorder=5,
               label="Surface (erasure-aware MWPM, $K=2$)")
    for x, y, lab in zip(ps_sc, ov_sc, lab_sc):
        ax.annotate(lab, (x, y), textcoords="offset points",
                    xytext=(4, 5), fontsize=7.5, color=SC_COLOR)

    ax.axvline(FSS_P_INF, color=FIT_COLOR, linestyle=":", linewidth=1.0,
               label=f"$p^*_\\infty = {FSS_P_INF:.3f}$", zorder=2)

    # Pareto annotation
    ax.annotate("", xy=(ps_bb[-1] + 0.001, ov_bb[-1]),
                xytext=(ps_sc[-1] - 0.001, ov_sc[-1]),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.1))
    ax.text((ps_bb[-1] + ps_sc[-1]) / 2 + 0.004,
            (ov_bb[-1] + ov_sc[-1]) / 2,
            f"12× less\noverhead", fontsize=7.5, ha="left", va="center",
            color="black")

    ax.set_yscale("log")
    ax.set_xlabel(r"Pseudo-threshold $p^*$", fontsize=11)
    ax.set_ylabel(r"Physical qubits per logical ($N/K$, log scale)", fontsize=11)
    ax.set_title(r"Overhead vs.\ threshold: BB codes vs.\ surface codes", fontsize=11)
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"[plot] Overhead vs threshold -> {out_path}")


# ---------------------------------------------------------------------------
# Figure 4 — Fraction of capacity: p* / (1 - K/N)  vs N
# ---------------------------------------------------------------------------
def plot_fraction_of_capacity(out_path: Path) -> None:
    Ns    = np.array([r[2] for r in BB])
    Ks    = np.array([r[3] for r in BB])
    ps    = np.array([r[4] for r in BB])
    ci_lo = np.array([r[5] for r in BB])
    ci_hi = np.array([r[6] for r in BB])
    labels = [f"${r[0]}\\times{r[1]}$" for r in BB]

    # Shannon limit for the erasure channel at rate K/N: p_hash = 1 - K/N
    p_hash  = 1.0 - Ks / Ns
    frac    = ps    / p_hash
    frac_lo = ci_lo / p_hash
    frac_hi = ci_hi / p_hash

    # Asymptotic: p*_inf / 1.0  (rate -> 0, p_hash -> 1)
    frac_inf = FSS_P_INF / 1.0

    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    ax.axhline(frac_inf, color=FIT_COLOR, linestyle="--", linewidth=1.1,
               label=f"$p^*_\\infty / 1 = {frac_inf:.3f}$  (FSS asymptote)", zorder=2)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.0,
               label="Zero-rate hashing bound ($p=0.5$)", zorder=1)

    yerr = [frac - frac_lo, frac_hi - frac]
    ax.errorbar(Ns, frac, yerr=yerr,
                fmt="o-", color=BB_COLOR, capsize=4, capthick=1.2,
                linewidth=1.3, zorder=5,
                label=r"$p^* \,/\, (1 - K/N)$")

    for x, y, lab in zip(Ns, frac, labels):
        ax.annotate(lab, (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=8, color=BB_COLOR)

    ax.set_xscale("log")
    ax.set_xlabel(r"Code size $N$ (log scale)", fontsize=11)
    ax.set_ylabel(r"$p^* \,/\, (1 - K/N)$  (fraction of Shannon limit)", fontsize=11)
    ax.set_title("BB code pseudo-threshold as fraction of Shannon limit", fontsize=11)
    ax.set_ylim(0.38, 0.52)
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"[plot] Fraction of capacity -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results", help="Output directory")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(PAPER_RC)

    plot_linearised_fss(out / "plot_fss_linear.pdf")
    plot_rate_vs_threshold(out / "plot_rate_vs_threshold.pdf")
    plot_overhead_threshold(out / "plot_overhead_threshold.pdf")
    plot_fraction_of_capacity(out / "plot_fraction_capacity.pdf")

    print("\nAll supplementary figures saved to", out)


if __name__ == "__main__":
    main()
