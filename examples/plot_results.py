#!/usr/bin/env python3
"""
Plot saved scaling results.

Reads one or many results/scaling_*.csv (and any available scaling_*.json) and produces:
  1) WER vs p curves grouped by (L,M)
  2) p_at_target_wer vs N (from JSON metadata threshold_estimates)

Outputs PNGs into results/ using stable filenames (no timestamp spam).

Examples:
  python examples/plot_results.py --all
  python examples/plot_results.py --csv results/scaling_20251224T171422Z_adaptive_010.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import matplotlib.pyplot as plt


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def maybe_matching_json(csv_path: Path) -> Optional[Path]:
    j = csv_path.with_suffix(".json")
    return j if j.exists() else None


def to_float(x: str) -> float:
    return float(x.strip())


def to_int(x: str) -> int:
    x = x.strip()
    return int(x) if x else 0


def wilson_ci_95(fails: int, shots: int) -> Tuple[float, float]:
    """
    Wilson score interval for a binomial proportion at ~95% confidence.
    Returns (lo, hi). If shots==0, returns (0,0).
    """
    if shots <= 0:
        return 0.0, 0.0
    z = 1.959963984540054  # ~95%
    n = float(shots)
    phat = float(fails) / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    half = (z / denom) * ((phat * (1.0 - phat) / n + (z * z) / (4.0 * n * n)) ** 0.5)
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo, hi


def _dedupe_points_keep_max_shots(
    points: List[Dict[str, str]],
) -> Dict[Tuple[int, int], List[Tuple[float, float, int, int]]]:
    """
    For each (L,M) and p, keep the row with the largest shot count.
    Returns: {(L,M): [(p, wer, fails, shots), ... sorted by p]}
    """
    by = defaultdict(dict)  # (L,M) -> p -> (shots, wer, fails)
    for r in points:
        L = to_int(r["L"])
        M = to_int(r["M"])
        p = float(r["p"])
        wer = float(r["wer"])
        shots = int(float(r.get("shots", "0") or 0))
        fails = int(float(r.get("fails", "0") or 0))
        cur = by[(L, M)].get(p)
        if cur is None or shots > cur[0]:
            by[(L, M)][p] = (shots, wer, fails)
    out = {}
    for key, mp in by.items():
        pts = sorted(((p, wer, fails, shots) for p, (shots, wer, fails) in mp.items()), key=lambda t: t[0])
        out[key] = pts
    return out


def _load_threshold_estimates_from_jsons(json_paths: List[Path]) -> List[Dict[str, Any]]:
    """
    Combine threshold_estimates from many JSON files.
    If multiple entries exist for the same (L,M,target_wer), keep the latest by file mtime.
    """
    best: Dict[Tuple[int, int, float], Tuple[float, Dict[str, Any]]] = {}
    for jp in json_paths:
        try:
            meta = load_json(jp)
        except Exception:
            continue
        ests = meta.get("threshold_estimates") or []
        mtime = jp.stat().st_mtime
        for e in ests:
            L = int(e.get("L"))
            M = int(e.get("M"))
            target = float(e.get("target_wer"))
            key = (L, M, target)
            if key not in best or mtime > best[key][0]:
                best[key] = (mtime, {"L": L, "M": M, "N": e.get("N"), "target_wer": target, "p_at_target_wer": e.get("p_at_target_wer")})
    return [v for _, v in best.values()]

def estimate_pstar_and_err_from_points(
    pts: List[Tuple[float, float, int, int]],
    target_wer: float,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Estimate p* where WER crosses target_wer using linear interpolation on the
    closest bracketing points. Also return an approximate 1-sigma uncertainty
    in p* by propagating binomial standard errors from the two bracketing points.
    """
    if not pts:
        return None, None
    pts_sorted = sorted(pts, key=lambda t: t[0])
    for (p0, w0, f0, n0), (p1, w1, f1, n1) in zip(pts_sorted, pts_sorted[1:]):
        if (w0 - target_wer) * (w1 - target_wer) <= 0 and w0 != w1:
            t = (target_wer - w0) / (w1 - w0)
            pstar = p0 + t * (p1 - p0)

            # Approx 1-sigma on W from binomial normal approx; then map to p via slope
            import math

            se0 = math.sqrt(max(w0 * (1.0 - w0), 1e-12) / max(n0, 1))
            se1 = math.sqrt(max(w1 * (1.0 - w1), 1e-12) / max(n1, 1))
            slope = abs((w1 - w0) / (p1 - p0))
            if slope <= 0:
                return pstar, None
            # Conservative: use the larger SE
            p_err = max(se0, se1) / slope
            return pstar, p_err
    return None, None


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot QLDPC scaling results from CSV/JSON.")
    parser.add_argument("--csv", type=str, default="", help="Path to a specific results CSV")
    parser.add_argument("--all", action="store_true", help="Plot using ALL results/scaling_*.csv in the out-dir")
    parser.add_argument("--out-dir", type=str, default="results", help="Output directory for plots (default: results)")
    parser.add_argument("--title", type=str, default="", help="Optional title prefix")
    parser.add_argument("--style", type=str, default="default", 
                       choices=["default", "paper", "presentation"],
                       help="Plot style preset (default: default)")
    parser.add_argument("--dpi", type=int, default=200, help="Output DPI for PNG (default: 200)")
    parser.add_argument("--format", type=str, default="png",
                       choices=["png", "pdf", "svg"],
                       help="Output format (default: png)")
    parser.add_argument("--figsize", type=str, default="8,5",
                       help="Figure size as width,height in inches (default: 8,5)")
    parser.add_argument("--no-error-bars", action="store_true",
                       help="Disable Wilson confidence interval error bars")
    args = parser.parse_args()

    # Parse figure size
    try:
        figw, figh = [float(x.strip()) for x in args.figsize.split(",")]
    except:
        figw, figh = 8, 5
    
    # Apply style presets
    if args.style == "paper":
        plt.rcParams.update({
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 8,
            'figure.dpi': args.dpi if args.format == "png" else 300,
            'lines.linewidth': 1.0,
            'lines.markersize': 4,
        })
    elif args.style == "presentation":
        plt.rcParams.update({
            'font.size': 14,
            'axes.labelsize': 16,
            'axes.titlesize': 18,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 11,
            'figure.dpi': args.dpi if args.format == "png" else 150,
            'lines.linewidth': 2.5,
            'lines.markersize': 6,
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pick CSV(s)
    csv_paths: List[Path]
    if args.csv:
        csv_paths = [Path(args.csv)]
    elif args.all:
        csv_paths = sorted(list(out_dir.glob("scaling_*.csv")) + list(out_dir.glob("baseline_surface_*.csv")), key=lambda p: p.stat().st_mtime)
        if not csv_paths:
            raise SystemExit("No results/scaling_*.csv or baseline_surface_*.csv found. Run a sweep first.")
    else:
        candidates = sorted(list(out_dir.glob("scaling_*.csv")) + list(out_dir.glob("baseline_surface_*.csv")), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise SystemExit("No results/scaling_*.csv or baseline_surface_*.csv found. Run a sweep first.")
        csv_paths = [candidates[0]]

    all_rows: List[Dict[str, str]] = []
    for pth in csv_paths:
        rows = load_csv(pth)
        all_rows.extend(rows)

    if not all_rows:
        raise SystemExit("No rows found in selected CSV file(s).")

    # Dedupe points per (L,M,p) using max shots (best statistics)
    by_size = _dedupe_points_keep_max_shots(all_rows)

    # Best-effort size info from rows (latest seen for each size)
    size_info: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for r in all_rows:
        L = to_int(r["L"])
        M = to_int(r["M"])
        size_info[(L, M)] = {"N": to_int(r.get("N", "")), "K": r.get("K", ""), "family": r.get("family", "bivariate_bicycle")}

    # ---- Plot 1: WER vs p curves ----
    plt.figure(figsize=(figw, figh))
    for (L, M), pts in sorted(by_size.items(), key=lambda kv: size_info[kv[0]].get("N", 0)):
        xs = [p for p, _, _, _ in pts]
        ys = [w for _, w, _, _ in pts]
        
        info = size_info[(L, M)]
        N = info.get("N", "")
        fam = info.get("family", "")
        label = f"{fam}: {L}x{M} (N={N})" if fam else f"{L}x{M} (N={N})"
        
        if args.no_error_bars:
            # Simple line plot without error bars
            plt.plot(xs, ys, '-o', linewidth=1.5, markersize=3, label=label)
        else:
            # 95% CI half-widths for error bars
            yerr_lo = []
            yerr_hi = []
            for _, w, fails, shots in pts:
                lo, hi = wilson_ci_95(int(fails), int(shots))
                yerr_lo.append(max(0.0, w - lo))
                yerr_hi.append(max(0.0, hi - w))
            plt.errorbar(
                xs,
                ys,
                yerr=[yerr_lo, yerr_hi],
                fmt="-o",
                linewidth=1.5,
                markersize=3,
                capsize=2,
                label=label,
            )

    title = args.title.strip()
    if not title:
        if args.all:
            title = "WER vs erasure rate (combined runs)"
        else:
            title = f"WER vs erasure rate (from {csv_paths[0].name})"
    plt.title(title)
    plt.xlabel("Erasure rate p")
    plt.ylabel("Word error rate (WER)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=1)
    plt.tight_layout()

    out1 = out_dir / f"plot_all_wer_vs_p.{args.format}" if args.all else out_dir / f"plot_wer_vs_p_{csv_paths[0].stem}.{args.format}"
    plt.savefig(out1, dpi=args.dpi if args.format == "png" else None, format=args.format)
    plt.close()

    # ---- Plot 2: p@target_WER vs N (from JSON) ----
    out2 = None
    json_paths = sorted(
        list(out_dir.glob("scaling_*.json")) + list(out_dir.glob("baseline_surface_*.json")),
        key=lambda p: p.stat().st_mtime,
    )
    estimates = _load_threshold_estimates_from_jsons(json_paths)
    # If not --all and a single csv was selected, prefer matching json only
    if not args.all and csv_paths:
        mj = maybe_matching_json(csv_paths[0])
        if mj and mj.exists():
            estimates = _load_threshold_estimates_from_jsons([mj])

    if estimates:
        # Group by target_wer (usually just one)
        by_target = defaultdict(list)
        for e in estimates:
            if e.get("p_at_target_wer") is None:
                continue
            by_target[float(e["target_wer"])].append(e)

        # Plot the first target_wer available
        target = sorted(by_target.keys())[0] if by_target else None
        if target is not None:
            pts_meta = sorted(by_target[target], key=lambda e: int(e.get("N") or 0))
            xsN = [int(e["N"]) for e in pts_meta if e.get("N") is not None]
            ysp = [float(e["p_at_target_wer"]) for e in pts_meta if e.get("p_at_target_wer") is not None]
            labels = [f"{e['L']}x{e['M']}" for e in pts_meta]

            # Approx p* uncertainty from bracketing points in the combined CSV data
            yerr = []
            for e in pts_meta:
                L = int(e["L"])
                M = int(e["M"])
                pstar, perr = estimate_pstar_and_err_from_points(by_size.get((L, M), []), target)
                yerr.append(0.0 if perr is None else float(perr))

            if xsN:
                plt.figure(figsize=(figw * 0.875, figh * 0.9))
                plt.errorbar(xsN, ysp, yerr=yerr, marker="o", linewidth=1.5, capsize=3)
                for x, y, lab in zip(xsN, ysp, labels):
                    plt.annotate(lab, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
                plt.title(f"p at target WER={target} vs N")
                plt.xlabel("N (physical qubits)")
                plt.ylabel("p@target_WER")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                out2 = out_dir / f"plot_all_pstar_vs_N.{args.format}" if args.all else out_dir / f"plot_pstar_vs_N_{csv_paths[0].stem}.{args.format}"
                plt.savefig(out2, dpi=args.dpi if args.format == "png" else None, format=args.format)
                plt.close()

    print(f"Saved: {out1}")
    if out2:
        print(f"Saved: {out2}")
    else:
        print("Note: No threshold_estimates found (or no p_at_target_wer values) in JSON; skipped p* vs N plot.")

    # ---- Plot 3: Overhead (data qubits per logical) vs N, with surface-code baseline ----
    # For rotated surface code memory patch: k=1 and data qubits N_sc = d^2. If we compare at the
    # same data-qubit count N, surface has overhead N per logical, while BB has N/K per logical.
    sizes_sorted = sorted(size_info.items(), key=lambda kv: kv[1].get("N", 0))
    xsN = []
    y_bb = []
    labs = []
    for (L, M), info in sizes_sorted:
        N = info.get("N", 0)
        K = info.get("K", "")
        if not N:
            continue
        try:
            K_int = int(K) if str(K).strip() else 0
        except Exception:
            K_int = 0
        if K_int <= 0:
            continue
        xsN.append(int(N))
        y_bb.append(float(N) / float(K_int))
        labs.append(f"{L}x{M} (K={K_int})")

    out3 = None
    if xsN:
        plt.figure(figsize=(figw * 0.9375, figh * 0.96))
        plt.plot(xsN, y_bb, marker="o", linewidth=1.5, label="Bivariate bicycle (data qubits per logical)")
        # Surface code baseline at same data-qubit count N: per-logical overhead = N (since k=1)
        plt.plot(xsN, xsN, linestyle="--", linewidth=1.5, label="Rotated surface code baseline (k=1 ⇒ N per logical)")
        for x, y, lab in zip(xsN, y_bb, labs):
            plt.annotate(lab, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
        plt.yscale("log")
        plt.title("Data qubits per logical vs N (surface baseline is k=1)")
        plt.xlabel("N (data/physical qubits)")
        plt.ylabel("Data qubits per logical (log scale)")
        plt.grid(True, which="both", alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        out3 = out_dir / f"plot_all_overhead_vs_surface.{args.format}" if args.all else out_dir / f"plot_overhead_vs_surface_{csv_paths[0].stem}.{args.format}"
        plt.savefig(out3, dpi=args.dpi if args.format == "png" else None, format=args.format)
        plt.close()

    if out3:
        print(f"Saved: {out3}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


