#!/usr/bin/env python3
"""
Run extended sweeps for all code sizes to find thresholds.

This script runs adaptive threshold sweeps for multiple code sizes,
extending the p-range to find WER=0.10 thresholds for larger codes.
All results are saved with consistent seeding for reproducibility.

Usage:
    python examples/run_all_extended_sweeps.py
    python examples/run_all_extended_sweeps.py --include-large  # Also run 30x15, 36x18
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def run_sweep(sizes: str, p_max: float, shots: int, seed: int, tag: str = ""):
    """Run a single sweep and return success status."""
    cmd = [
        sys.executable,
        "examples/sweep_scaling.py",
        "--sizes", sizes,
        "--adaptive",
        "--shots", str(shots),
        "--seed", str(seed),
        "--p-start", "0.30",
        "--p-max", str(p_max),
        "--bracket-step", "0.02",
        "--refine-steps", "6",
    ]
    if tag:
        cmd.extend(["--tag", tag])
    
    print(f"\n{'='*70}")
    print(f"Running sweep: {sizes} (p_max={p_max}, shots={shots}, seed={seed})")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    
    start_time = datetime.now()
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\n✓ Completed in {elapsed/60:.1f} minutes\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed with exit code {e.returncode}\n")
        return False
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user\n")
        return False


def regenerate_plots(style: str = "paper", format: str = "pdf", dpi: int = 300):
    """Regenerate all plots from the latest results."""
    cmd = [
        sys.executable,
        "examples/plot_results.py",
        "--all",
        "--style", style,
        "--format", format,
        "--dpi", str(dpi),
    ]
    
    print(f"\n{'='*70}")
    print(f"Regenerating plots (style={style}, format={format}, dpi={dpi})")
    print(f"{'='*70}\n")
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✓ Plots regenerated successfully\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Plot generation failed with exit code {e.returncode}\n")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run extended sweeps for all code sizes to find thresholds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run extended sweeps for 18x9 and 24x12 only
  python examples/run_all_extended_sweeps.py

  # Include larger codes (30x15, 36x18) - takes much longer
  python examples/run_all_extended_sweeps.py --include-large

  # Custom shot count and p-max
  python examples/run_all_extended_sweeps.py --shots 100000 --p-max 0.50
        """
    )
    parser.add_argument(
        "--include-large",
        action="store_true",
        help="Also run sweeps for 30x15 and 36x18 (takes 6-12 hours total)",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=200000,
        help="Shots per point (default: 200000)",
    )
    parser.add_argument(
        "--p-max",
        type=float,
        default=0.55,
        help="Maximum erasure rate to test (default: 0.55)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for reproducibility (default: 12345)",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot regeneration at the end",
    )
    parser.add_argument(
        "--plot-format",
        type=str,
        default="pdf",
        choices=["png", "pdf", "svg"],
        help="Plot output format (default: pdf)",
    )
    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=300,
        help="Plot DPI (default: 300)",
    )
    
    args = parser.parse_args()
    
    # Define sweep batches
    sweeps = [
        ("18x9,24x12", "extended"),
    ]
    
    if args.include_large:
        sweeps.append(("30x15,36x18", "extended_large"))
    
    print("\n" + "="*70)
    print("EXTENDED THRESHOLD SWEEP RUNNER")
    print("="*70)
    print(f"Configuration:")
    print(f"  Shots per point: {args.shots:,}")
    print(f"  Max erasure rate: {args.p_max}")
    print(f"  Seed: {args.seed}")
    print(f"  Code sizes: {', '.join([s[0] for s in sweeps])}")
    print(f"  Total sweeps: {len(sweeps)}")
    print("="*70)
    
    # Estimate runtime
    if args.include_large:
        print("\n⚠️  WARNING: Including large codes (30x15, 36x18) will take 6-12 hours total.")
        print("   Consider running without --include-large first to verify everything works.")
        response = input("\nContinue? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled.")
            return 0
    
    # Run sweeps
    start_time = datetime.now()
    success_count = 0
    
    for sizes, tag in sweeps:
        success = run_sweep(
            sizes=sizes,
            p_max=args.p_max,
            shots=args.shots,
            seed=args.seed,
            tag=tag,
        )
        if success:
            success_count += 1
        else:
            print(f"\n⚠️  Warning: Sweep for {sizes} failed. Continuing with remaining sweeps...\n")
    
    total_elapsed = (datetime.now() - start_time).total_seconds()
    
    # Summary
    print("\n" + "="*70)
    print("SWEEP SUMMARY")
    print("="*70)
    print(f"Completed: {success_count}/{len(sweeps)} sweeps")
    print(f"Total time: {total_elapsed/3600:.2f} hours ({total_elapsed/60:.1f} minutes)")
    print("="*70)
    
    # Regenerate plots
    if not args.skip_plots and success_count > 0:
        print("\nRegenerating plots from all results...")
        regenerate_plots(
            style="paper",
            format=args.plot_format,
            dpi=args.plot_dpi,
        )
        
        # Also generate PNG for quick viewing
        if args.plot_format != "png":
            regenerate_plots(
                style="paper",
                format="png",
                dpi=args.plot_dpi,
            )
    
    if success_count == len(sweeps):
        print("\n✓ All sweeps completed successfully!")
        print("\nNext steps:")
        print("  1. Check results/ directory for CSV/JSON files")
        print("  2. Review plots in results/plot_*.pdf")
        print("  3. Use plot_results.py to generate custom plots")
        return 0
    else:
        print(f"\n⚠️  {len(sweeps) - success_count} sweep(s) failed. Check output above for details.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

