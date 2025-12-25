#!/usr/bin/env python3
"""
Example script: Run Bivariate Bicycle Code simulation

This script demonstrates how to use the QLDPC framework to simulate
quantum error correction and determine error thresholds.
"""

import sys
import argparse
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qldpc import BivariateBicycleCode, QuantumSimulator, DecoderConfig


def _parse_rates_csv(text: str):
    """
    Parse a comma-separated list of floats, e.g. "0.1,0.12,0.2".
    """
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return [float(p) for p in parts]


def _frange(start: float, stop: float, step: float):
    """
    Float range, inclusive of stop within numerical tolerance.
    """
    if step <= 0:
        raise ValueError("--step must be > 0")
    values = []
    x = start
    # Include stop (within tolerance) to avoid 0.399999999 issues
    while x <= stop + 1e-12:
        values.append(round(x, 10))
        x += step
    return values


def main():
    """Run the main simulation experiment."""
    parser = argparse.ArgumentParser(
        description="Run Bivariate Bicycle Code simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_simulation.py                    # Use default 5000 shots
  python run_simulation.py --shots 50000     # Use 50,000 shots
  python run_simulation.py -s 100000         # Use 100,000 shots
  python run_simulation.py --high-threshold --max-rate 0.40
  python run_simulation.py --rates 0.14,0.18,0.22,0.26,0.30,0.34,0.38,0.40
        """
    )
    parser.add_argument(
        "-s", "--shots",
        type=int,
        default=5000,
        help="Total number of simulation shots per erasure rate (default: 5000)"
    )
    parser.add_argument(
        "--high-threshold",
        action="store_true",
        help="Test higher erasure rates to find threshold (0.14, 0.16, 0.18, 0.20)"
    )
    parser.add_argument(
        "--max-rate",
        type=float,
        default=0.24,
        help="Max erasure rate for --high-threshold sweep (default: 0.24)"
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.02,
        help="Step size for --high-threshold sweep (default: 0.02)"
    )
    parser.add_argument(
        "--rates",
        type=_parse_rates_csv,
        default=None,
        help="Comma-separated erasure rates to test (overrides presets), e.g. --rates 0.1,0.12,0.2"
    )
    
    args = parser.parse_args()
    
    # Setup Code
    code = BivariateBicycleCode(L=12, M=6)
    
    # Configure Decoder (optional - uses defaults if not specified)
    config = DecoderConfig(
        bp_method="min_sum",
        osd_method="osd_cs",
        osd_order=10,
        max_iter=50
    )
    
    # Create Simulator
    simulator = QuantumSimulator(code, config=config)
    
    # Run Experiment
    if args.rates is not None:
        erasure_rates = args.rates
    elif args.high_threshold:
        erasure_rates = _frange(0.14, args.max_rate, args.step)
    else:
        erasure_rates = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
    total_shots = args.shots
    
    results = simulator.run_experiment(
        erasure_rates=erasure_rates,
        total_shots=total_shots,
        verbose=True
    )
    
    # Print summary
    print("\n=== RESULTS SUMMARY ===")
    for p, wer in results.items():
        print(f"Erasure Rate: {p:.4f} -> WER: {wer:.5f}")
    
    return results


if __name__ == "__main__":
    main()

