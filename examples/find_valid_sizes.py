#!/usr/bin/env python3
"""
Find (L, M) pairs that yield non-trivial codes (K > 0) for the current
BivariateBicycleCode polynomial choice.

This is important because many (L, M) choices can produce K=0, in which case
"logical error rate" is not meaningful.

Example:
  python examples/find_valid_sizes.py --L 6 40 --M 3 20 --min-K 1 --top 30

Or scan a fixed aspect ratio L=2M:
  python examples/find_valid_sizes.py --ratio 2 --M 3 30 --min-K 1
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qldpc import BivariateBicycleCode, DecoderConfig
from qldpc.decoder import create_decoder


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan (L,M) for K>0 under current BB code definition.")

    parser.add_argument("--min-K", type=int, default=1, help="Only report sizes with K >= min-K (default: 1)")
    parser.add_argument("--top", type=int, default=0, help="Only print first N matches (0 = all)")

    # Option A: rectangular range
    parser.add_argument("--L", nargs=2, type=int, metavar=("L_MIN", "L_MAX"), help="Scan L in [L_MIN, L_MAX]")
    parser.add_argument("--M", nargs=2, type=int, metavar=("M_MIN", "M_MAX"), help="Scan M in [M_MIN, M_MAX]")

    # Option B: fixed ratio L = ratio * M, scan M range
    parser.add_argument("--ratio", type=int, default=0, help="If set, scan sizes with L = ratio*M")

    args = parser.parse_args()

    config = DecoderConfig()

    matches = 0

    if args.ratio:
        if not args.M:
            raise SystemExit("When using --ratio, you must also provide --M M_MIN M_MAX")
        m_min, m_max = args.M
        for M in range(m_min, m_max + 1):
            L = args.ratio * M
            code = BivariateBicycleCode(L=L, M=M)
            Hx, Hz = code.get_matrices()
            qcode, _ = create_decoder(Hx, Hz, config, initial_error_rate=0.1)
            K = getattr(qcode, "K", None)
            if K is None:
                continue
            if K >= args.min_K:
                matches += 1
                N = 2 * L * M
                print(f"L={L:>3} M={M:>3}  N={N:>4}  K={int(K):>4}   (Hx={Hx.shape[0]}x{Hx.shape[1]})")
                if args.top and matches >= args.top:
                    break
        return 0

    if not (args.L and args.M):
        raise SystemExit("Provide either (--L L_MIN L_MAX and --M M_MIN M_MAX) or (--ratio and --M).")

    l_min, l_max = args.L
    m_min, m_max = args.M

    for L in range(l_min, l_max + 1):
        for M in range(m_min, m_max + 1):
            code = BivariateBicycleCode(L=L, M=M)
            Hx, Hz = code.get_matrices()
            qcode, _ = create_decoder(Hx, Hz, config, initial_error_rate=0.1)
            K = getattr(qcode, "K", None)
            if K is None:
                continue
            if K >= args.min_K:
                matches += 1
                N = 2 * L * M
                print(f"L={L:>3} M={M:>3}  N={N:>4}  K={int(K):>4}   (Hx={Hx.shape[0]}x{Hx.shape[1]})")
                if args.top and matches >= args.top:
                    return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


