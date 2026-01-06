#!/usr/bin/env python3
"""
Compare two or more simulation runs side-by-side.

This utility loads multiple CSV files and generates comparison plots showing
how different configurations (decoder settings, code families, etc.) perform.

Examples:
    # Compare OSD orders
    python examples/compare_runs.py \\
        --csv1 results/scaling_osd10.csv \\
        --csv2 results/scaling_osd15.csv \\
        --label1 "OSD-10" \\
        --label2 "OSD-15"
    
    # Compare code families
    python examples/compare_runs.py \\
        --csv1 results/scaling_bicycle.csv \\
        --csv2 results/baseline_surface.csv \\
        --label1 "Bivariate Bicycle" \\
        --label2 "Surface Code (MWPM)"
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def group_by_size(rows: List[Dict[str, str]]) -> Dict[Tuple[int, int], List[Tuple[float, float]]]:
    """Group points by (L, M) and return (p, wer) lists."""
    by_size = defaultdict(list)
    for r in rows:
        L = int(r["L"])
        M = int(r["M"])
        p = float(r["p"])
        wer = float(r["wer"])
        by_size[(L, M)].append((p, wer))
    # Sort each by p
    return {k: sorted(v, key=lambda t: t[0]) for k, v in by_size.items()}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare multiple simulation runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--csv1", required=True, help="First CSV file")
    parser.add_argument("--csv2", required=True, help="Second CSV file")
    parser.add_argument("--csv3", default="", help="Optional third CSV file")
    parser.add_argument("--csv4", default="", help="Optional fourth CSV file")
    
    parser.add_argument("--label1", default="Run 1", help="Label for first run")
    parser.add_argument("--label2", default="Run 2", help="Label for second run")
    parser.add_argument("--label3", default="Run 3", help="Label for third run")
    parser.add_argument("--label4", default="Run 4", help="Label for fourth run")
    
    parser.add_argument("--out", default="results/comparison.png", help="Output plot path")
    parser.add_argument("--dpi", type=int, default=200, help="Output DPI")
    parser.add_argument("--title", default="", help="Plot title (default: auto)")
    
    args = parser.parse_args()
    
    # Load CSV files
    csv_paths = [Path(args.csv1), Path(args.csv2)]
    labels = [args.label1, args.label2]
    
    if args.csv3:
        csv_paths.append(Path(args.csv3))
        labels.append(args.label3)
    if args.csv4:
        csv_paths.append(Path(args.csv4))
        labels.append(args.label4)
    
    # Load and group data
    all_data = []
    for p in csv_paths:
        if not p.exists():
            print(f"Error: {p} not found")
            return 1
        rows = load_csv(p)
        by_size = group_by_size(rows)
        all_data.append(by_size)
    
    # Find common sizes
    common_sizes = set(all_data[0].keys())
    for d in all_data[1:]:
        common_sizes &= set(d.keys())
    
    if not common_sizes:
        print("Error: No common code sizes found across all runs")
        return 1
    
    common_sizes = sorted(common_sizes, key=lambda t: t[0] * t[1])  # Sort by N
    
    # Create subplots (one per size)
    n_sizes = len(common_sizes)
    n_cols = min(3, n_sizes)
    n_rows = (n_sizes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_sizes == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes
    
    colors = ['C0', 'C1', 'C2', 'C3']
    markers = ['o', 's', '^', 'D']
    
    for idx, (L, M) in enumerate(common_sizes):
        ax = axes[idx]
        N = 2 * L * M
        
        for run_idx, (data, label) in enumerate(zip(all_data, labels)):
            pts = data[(L, M)]
            xs = [p for p, _ in pts]
            ys = [w for _, w in pts]
            ax.plot(xs, ys, 
                   color=colors[run_idx], 
                   marker=markers[run_idx],
                   label=label,
                   linewidth=1.5,
                   markersize=4)
        
        ax.set_title(f"L={L}, M={M}, N={N}")
        ax.set_xlabel("Erasure rate p")
        ax.set_ylabel("Word error rate (WER)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    # Hide unused subplots
    for idx in range(n_sizes, len(axes)):
        axes[idx].axis('off')
    
    title = args.title if args.title else f"Comparison: {' vs '.join(labels)}"
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=args.dpi)
    print(f"Saved comparison plot: {out_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

