#!/usr/bin/env python3
"""
Phase 1 - Mixed erasure + depolarizing channel sweep.

For each (L, M) code size and delta/epsilon ratio, runs an adaptive threshold
sweep over erasure rate p, where depolar_rate = (delta/eps) * p each shot.

Two-phase shots strategy (saves ~4x runtime vs. running 200k everywhere):
  Bracketing phase  : OSD-5,  30k shots - fast, discovers threshold region
  Refinement phase  : OSD-10, 200k shots - precise, evaluates near threshold

delta/eps ratios surveyed: [0.0, 0.1, 0.3, 0.5, 0.75, 1.0]
  delta/eps = 0.0 -> pure erasure (reproduces original results as a sanity check)
  delta/eps ~ 0.3-0.5 matches current neutral-atom hardware (Evered et al., 2023)

CSV output columns:
  family, L, M, N, K, p, shots, fails, wer, seconds,
  channel_model, depolar_rate, delta_eps_ratio

Usage examples:
  # Quick test on Gross code only, two delta/eps values
  python examples/sweep_mixed_channel.py --sizes 12x6 --delta-eps 0.0,0.3

  # Full Phase 1 sweep (all 5 sizes x 6 delta/eps values)
  python examples/sweep_mixed_channel.py

  # Override seed and output directory
  python examples/sweep_mixed_channel.py --seed 42 --out-dir results/phase1
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qldpc import BivariateBicycleCode, ChannelConfig, DecoderConfig, QuantumSimulator

# ---------------------------------------------------------------------------
# Default sweep parameters
# ---------------------------------------------------------------------------
DEFAULT_SIZES: List[Tuple[int, int]] = [
    (12, 6),   # [[144, 12, 12]]  Gross code
    (18, 9),   # [[324,  8]]
    (24, 12),  # [[576, 16]]
    (30, 15),  # [[900,  8]]
    (36, 18),  # [[1296, 12]]
]
DEFAULT_DELTA_EPS: List[float] = [0.0, 0.1, 0.3, 0.5, 0.75, 1.0]

# Two-pass shot counts
SHOTS_BRACKET = 30_000     # Fast: OSD-5, 30k - bracket the threshold
SHOTS_REFINE  = 200_000    # Precise: OSD-10, 200k - evaluated during bisection

# Adaptive bracketing parameters
P_START       = 0.30
P_STEP_COARSE = 0.03
P_MAX         = 0.50
BISECT_STEPS  = 6
TARGET_WER    = 0.10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_sizes(text: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for part in text.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if "x" not in part:
            raise ValueError(f"Bad size '{part}'. Expected e.g. 12x6")
        a, b = part.split("x", 1)
        out.append((int(a), int(b)))
    if not out:
        raise ValueError("No sizes provided")
    return out


def parse_delta_eps(text: str) -> List[float]:
    out: List[float] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        v = float(part)
        if not (0.0 <= v <= 5.0):
            raise ValueError(f"delta/eps value {v} is out of range [0, 5]")
        out.append(v)
    if not out:
        raise ValueError("No delta/eps values provided")
    return out


def safe_pkg_version(name: str) -> Optional[str]:
    try:
        from importlib import metadata
        return metadata.version(name)
    except Exception:
        try:
            import pkg_resources
            return pkg_resources.get_distribution(name).version
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Adaptive threshold sweep
# ---------------------------------------------------------------------------

def make_channel_cfg(delta_eps: float, p: float) -> ChannelConfig:
    """Return ChannelConfig for this (delta/eps, p) operating point."""
    return ChannelConfig(depolar_rate=delta_eps * p)


def adaptive_sweep(
    sim: QuantumSimulator,
    delta_eps: float,
    pool_fast,
    pool_prec,
    seed: Optional[int],
    shots_bracket: int = SHOTS_BRACKET,
    shots_refine: int = SHOTS_REFINE,
    p_start: float = P_START,
    p_step: float = P_STEP_COARSE,
    p_max: float = P_MAX,
    bisect_steps: int = BISECT_STEPS,
    target_wer: float = TARGET_WER,
    verbose: bool = True,
) -> Tuple[List[dict], Optional[float]]:
    """
    Adaptive threshold search with two-pass shots strategy.

    Phase A  -- OSD-fast pool, shots_bracket shots: step p from p_start in
                steps of p_step until WER crosses target_wer (brackets the
                threshold).
    Phase B  -- OSD-prec pool, shots_refine shots: bisect the bracket
                bisect_steps times for a precise p* estimate.

    Returns:
      rows   - list of result dicts (one per evaluated p-value)
      p_star - interpolated p at target_wer (or None if threshold not found)
    """
    rows: List[dict] = []
    cache: dict = {}

    def eval_p(p: float, shots: int, pool) -> dict:
        """Run one p-value; return cached result if already evaluated at >= shots."""
        p = round(p, 9)
        cached = cache.get(p)
        if cached is not None and cached["shots"] >= shots:
            return cached
        ch_cfg = make_channel_cfg(delta_eps, p)
        r = sim.run_point_prepared(
            erasure_rate=p,
            total_shots=shots,
            pool=pool,
            seed=seed,
            channel_config=ch_cfg,
        )
        row = {
            "p": p,
            "wer": r["wer"],
            "fails": int(r["fails"]),
            "shots": int(r["shots"]),
            "seconds": float(r["seconds"]),
            "depolar_rate": ch_cfg.depolar_rate,
            "channel_model": ch_cfg.to_dict()["channel_model"],
        }
        cache[p] = row
        rows.append(row)
        if verbose:
            print(
                f"    p={p:.5f}  wer={row['wer']:.5f}  "
                f"fails={row['fails']}  shots={row['shots']}  "
                f"depolar={row['depolar_rate']:.5f}  sec={row['seconds']:.1f}",
                flush=True,
            )
        return row

    # ---- Phase A: coarse bracketing (fast pool, fewer shots) ----
    p_lo = p_hi = None
    p = p_start
    while p <= p_max + 1e-9:
        r = eval_p(p, shots_bracket, pool_fast)
        if r["wer"] < target_wer:
            p_lo = p
            p += p_step
            continue
        if p_lo is None:
            return rows, None   # started above target, can't bracket
        p_hi = p
        break
    else:
        return rows, None   # never crossed

    if verbose:
        print(
            f"  Bracket: p_lo={p_lo:.5f}  p_hi={p_hi:.5f} -- "
            f"bisecting {bisect_steps}x at {shots_refine:,} shots...",
            flush=True,
        )

    # ---- Phase B: bisection refinement (precise pool, full shots) ----
    r_lo = eval_p(p_lo, shots_refine, pool_prec)
    r_hi = eval_p(p_hi, shots_refine, pool_prec)
    w_lo, w_hi = r_lo["wer"], r_hi["wer"]

    for _ in range(bisect_steps):
        mid = round(0.5 * (p_lo + p_hi), 9)
        r_mid = eval_p(mid, shots_refine, pool_prec)
        if r_mid["wer"] < target_wer:
            p_lo, w_lo = mid, r_mid["wer"]
        else:
            p_hi, w_hi = mid, r_mid["wer"]

    if w_hi == w_lo:
        p_star = float(p_hi)
    else:
        t = (target_wer - w_lo) / (w_hi - w_lo)
        p_star = float(p_lo + t * (p_hi - p_lo))

    return rows, p_star


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def load_completed_combos(csv_path: Path, shots_refine: int) -> set:
    """
    Return the set of (L, M, delta_eps_ratio) tuples that are already
    complete in csv_path.

    A combo is considered complete when it has at least one row whose
    'shots' value is >= shots_refine (meaning the bisection phase ran).
    Combos that were only bracketed (shots == shots_bracket) are NOT
    considered complete and will be re-run.
    """
    done = set()
    if not csv_path.exists():
        return done
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if int(row["shots"]) >= shots_refine:
                    key = (int(row["L"]), int(row["M"]), float(row["delta_eps_ratio"]))
                    done.add(key)
            except (KeyError, ValueError):
                continue
    return done


def find_latest_csv(out_dir: Path) -> Optional[Path]:
    """Return the most recently modified mixed_channel_*.csv in out_dir, or None."""
    candidates = sorted(
        out_dir.glob("mixed_channel_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_existing_rows(csv_path: Path) -> List[dict]:
    """Load all rows from an existing CSV into a list of dicts."""
    if not csv_path.exists():
        return []
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Mixed erasure + depolarizing channel sweep (Phase 1).\n\n"
            "Typical workflow:\n"
            "  # Run one combo, stop:\n"
            "  python sweep_mixed_channel.py --sizes 12x6 --delta-eps 0.0 --out-dir results/phase1\n\n"
            "  # Continue from where you left off (auto-detects latest CSV):\n"
            "  python sweep_mixed_channel.py --resume --out-dir results/phase1\n\n"
            "  # Or point to a specific CSV:\n"
            "  python sweep_mixed_channel.py --resume --csv results/phase1/mixed_channel_XYZ.csv"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- What to run ---
    parser.add_argument(
        "--sizes",
        type=parse_sizes,
        default=DEFAULT_SIZES,
        metavar="LxM[,LxM,...]",
        help="Code sizes to sweep, e.g. 12x6,18x9 (default: all 5 BB sizes)",
    )
    parser.add_argument(
        "--delta-eps",
        type=parse_delta_eps,
        default=DEFAULT_DELTA_EPS,
        metavar="RATIO[,RATIO,...]",
        help=(
            "delta/eps ratios to sweep, e.g. 0.0,0.3,0.5 "
            "(default: 0,0.1,0.3,0.5,0.75,1.0)"
        ),
    )

    # --- Resume support ---
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip already-completed (L,M,delta_eps) combos. "
            "Auto-detects the latest CSV in --out-dir, or use --csv to specify one."
        ),
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Explicit path to the output CSV. "
            "With --resume, existing rows are loaded and completed combos are skipped. "
            "Without --resume, a fresh timestamped file is created in --out-dir."
        ),
    )

    # --- Decoder / shot counts ---
    parser.add_argument(
        "--shots-bracket",
        type=int,
        default=SHOTS_BRACKET,
        help=f"Shots during OSD-fast coarse bracketing (default: {SHOTS_BRACKET:,})",
    )
    parser.add_argument(
        "--shots-refine",
        type=int,
        default=SHOTS_REFINE,
        help=f"Shots during OSD-precise bisection (default: {SHOTS_REFINE:,})",
    )
    parser.add_argument(
        "--osd-order-fast",
        type=int,
        default=5,
        help="OSD order for bracketing pool (default: 5)",
    )
    parser.add_argument(
        "--osd-order-prec",
        type=int,
        default=10,
        help="OSD order for refinement pool (default: 10)",
    )
    parser.add_argument(
        "--bp-method",
        type=str,
        default="min_sum",
        help="BP method (default: min_sum)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=50,
        help="Max BP iterations (default: 50)",
    )

    # --- Adaptive sweep tuning ---
    parser.add_argument(
        "--bisect-steps",
        type=int,
        default=BISECT_STEPS,
        help=f"Bisection steps in refinement phase (default: {BISECT_STEPS})",
    )
    parser.add_argument(
        "--p-start",
        type=float,
        default=P_START,
        help=f"Starting erasure rate (default: {P_START})",
    )
    parser.add_argument(
        "--p-step",
        type=float,
        default=P_STEP_COARSE,
        help=f"Coarse bracketing step (default: {P_STEP_COARSE})",
    )
    parser.add_argument(
        "--p-max",
        type=float,
        default=P_MAX,
        help=f"Maximum erasure rate (default: {P_MAX})",
    )

    # --- Infrastructure ---
    parser.add_argument(
        "--cores",
        type=int,
        default=None,
        help="CPU cores per pool (default: cpu_count()-1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Base RNG seed (default: 12345)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/phase1",
        help="Output directory (default: results/phase1)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional suffix appended to auto-generated filenames",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-point progress output",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the list of combos to run (or skip) and exit without simulating.",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Determine CSV / JSON paths and load existing data if resuming       #
    # ------------------------------------------------------------------ #
    fieldnames = [
        "family", "L", "M", "N", "K",
        "p", "shots", "fails", "wer", "seconds",
        "channel_model", "depolar_rate", "delta_eps_ratio",
    ]

    if args.resume:
        # Find existing CSV to resume from
        if args.csv:
            csv_path = Path(args.csv)
        else:
            csv_path = find_latest_csv(out_dir)
            if csv_path is None:
                print(
                    f"[ERROR] --resume specified but no mixed_channel_*.csv found in {out_dir}.\n"
                    "Run without --resume to start a fresh sweep."
                )
                return 1
        meta_path = csv_path.with_suffix(".json")
        print(f"[RESUME] Loading existing results from: {csv_path}")

        # Load completed combos and all existing rows
        done_combos = load_completed_combos(csv_path, args.shots_refine)
        all_rows = load_existing_rows(csv_path)
        # Cast numeric fields back from strings
        for row in all_rows:
            for k in ("L", "M", "N", "K", "shots", "fails"):
                if row.get(k, "") != "":
                    row[k] = int(row[k])
            for k in ("p", "wer", "seconds", "depolar_rate", "delta_eps_ratio"):
                if row.get(k, "") != "":
                    row[k] = float(row[k])

        print(f"  Found {len(done_combos)} completed (L,M,delta_eps) combo(s):")
        for (L, M, de) in sorted(done_combos):
            print(f"    L={L} M={M} delta_eps={de:.3f}  [SKIP]")

        # Load metadata if it exists
        try:
            with meta_path.open(encoding="utf-8") as f:
                meta = json.load(f)
        except FileNotFoundError:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            meta = {"timestamp_utc": stamp, "threshold_estimates": []}

    else:
        done_combos = set()
        all_rows = []
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        tag = f"_{args.tag}" if args.tag else ""
        if args.csv:
            csv_path = Path(args.csv)
        else:
            csv_path = out_dir / f"mixed_channel_{stamp}{tag}.csv"
        meta_path = csv_path.with_suffix(".json")
        meta = {
            "timestamp_utc": stamp,
            "tag": args.tag,
            "description": "Phase 1 mixed erasure + depolarizing channel sweep",
            "threshold_estimates": [],
        }

    # Always refresh these metadata fields to match current run args
    meta.update({
        "delta_eps_ratios": args.delta_eps,
        "sizes": [{"L": L, "M": M, "N": 2 * L * M} for (L, M) in args.sizes],
        "shots_bracket": args.shots_bracket,
        "shots_refine": args.shots_refine,
        "osd_order_fast": args.osd_order_fast,
        "osd_order_prec": args.osd_order_prec,
        "bisect_steps": args.bisect_steps,
        "p_start": args.p_start,
        "p_step_coarse": args.p_step,
        "p_max": args.p_max,
        "target_wer": TARGET_WER,
        "seed": args.seed,
        "num_cores": args.cores or max(1, multiprocessing.cpu_count() - 1),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "package_versions": {
            "numpy": safe_pkg_version("numpy"),
            "scipy": safe_pkg_version("scipy"),
            "ldpc": safe_pkg_version("ldpc"),
            "bposd": safe_pkg_version("bposd"),
        },
    })
    meta.setdefault("threshold_estimates", [])

    num_cores = meta["num_cores"]

    # ------------------------------------------------------------------ #
    # Build the work queue                                                 #
    # ------------------------------------------------------------------ #
    todo = []
    for (L, M) in args.sizes:
        for de in args.delta_eps:
            key = (L, M, de)
            if key in done_combos:
                continue
            todo.append(key)

    if args.dry_run or not todo:
        total = len(args.sizes) * len(args.delta_eps)
        print(f"\nWork queue: {len(todo)} combo(s) to run, "
              f"{len(done_combos)} already done (of {total} total).")
        for (L, M, de) in todo:
            print(f"  L={L} M={M} delta_eps={de:.3f}")
        if not todo:
            print("Nothing to do. All combos are complete.")
        return 0

    print(f"\nWork queue: {len(todo)} combo(s) to run.")

    # ------------------------------------------------------------------ #
    # Decoder configs                                                      #
    # ------------------------------------------------------------------ #
    config_fast = DecoderConfig(
        bp_method=args.bp_method,
        osd_method="osd_cs",
        osd_order=args.osd_order_fast,
        max_iter=args.max_iter,
    )
    config_prec = DecoderConfig(
        bp_method=args.bp_method,
        osd_method="osd_cs",
        osd_order=args.osd_order_prec,
        max_iter=args.max_iter,
    )

    verbose = not args.quiet

    # ------------------------------------------------------------------ #
    # Main sweep loop                                                      #
    # ------------------------------------------------------------------ #
    # Group todo items by code size to avoid re-creating pools per delta_eps
    from itertools import groupby
    from operator import itemgetter

    # Build ordered list of sizes that have at least one todo item
    seen_sizes = []
    for (L, M, _) in todo:
        if (L, M) not in seen_sizes:
            seen_sizes.append((L, M))

    for (L, M) in seen_sizes:
        code = BivariateBicycleCode(L=L, M=M)
        N = int(2 * L * M)

        try:
            # Fast K computation via rank — avoids expensive decoder init in main process
            from ldpc.mod2 import rank as _rank
            hx_arr = code.hx.toarray() if hasattr(code.hx, "toarray") else code.hx
            hz_arr = code.hz.toarray() if hasattr(code.hz, "toarray") else code.hz
            K = int(N - _rank(hx_arr) - _rank(hz_arr))
        except Exception:
            K = None

        if K == 0:
            print(f"\n[SKIP] L={L} M={M} N={N}: K=0.")
            continue

        print(f"\n{'='*65}")
        print(f"CODE  L={L}  M={M}  N={N}  K={K}")
        print(f"{'='*65}")

        sim_fast = QuantumSimulator(code, config=config_fast, num_cores=num_cores, seed=args.seed)
        sim_prec = QuantumSimulator(code, config=config_prec, num_cores=num_cores, seed=args.seed)
        pool_fast, _ = sim_fast.make_prepared_pool()
        pool_prec, _ = sim_prec.make_prepared_pool()

        try:
            # All delta_eps values for this (L, M) that still need running
            delta_eps_todo = [de for (ll, mm, de) in todo if ll == L and mm == M]

            for delta_eps in delta_eps_todo:
                print(
                    f"\n  delta/eps = {delta_eps:.3f}  "
                    f"(depolar_rate = delta_eps * p at each p-value)",
                    flush=True,
                )
                rows, p_star = adaptive_sweep(
                    sim=sim_prec,
                    delta_eps=delta_eps,
                    pool_fast=pool_fast,
                    pool_prec=pool_prec,
                    seed=args.seed,
                    shots_bracket=args.shots_bracket,
                    shots_refine=args.shots_refine,
                    p_start=args.p_start,
                    p_step=args.p_step,
                    p_max=args.p_max,
                    bisect_steps=args.bisect_steps,
                    verbose=verbose,
                )

                status = f"p* ~ {p_star:.5f}" if p_star is not None else "threshold not found in range"
                print(f"  [{status}]", flush=True)

                meta["threshold_estimates"].append({
                    "L": L, "M": M, "N": N, "K": K,
                    "delta_eps_ratio": delta_eps,
                    "p_star": p_star,
                    "target_wer": TARGET_WER,
                })

                for r in rows:
                    all_rows.append({
                        "family": "bivariate_bicycle",
                        "L": L, "M": M, "N": N,
                        "K": "" if K is None else K,
                        "p": r["p"],
                        "shots": r["shots"],
                        "fails": r["fails"],
                        "wer": r["wer"],
                        "seconds": r["seconds"],
                        "channel_model": r["channel_model"],
                        "depolar_rate": r["depolar_rate"],
                        "delta_eps_ratio": delta_eps,
                    })

                # Flush after each completed combo so Ctrl-C loses at most one combo
                _write_csv(csv_path, fieldnames, all_rows)
                _write_meta(meta_path, meta)
                print(f"  [Saved -> {csv_path.name}]", flush=True)

        finally:
            pool_fast.close(); pool_fast.join()
            pool_prec.close(); pool_prec.join()

    # Final write (redundant but safe)
    _write_csv(csv_path, fieldnames, all_rows)
    _write_meta(meta_path, meta)

    print(f"\n{'='*65}")
    print(f"Saved CSV  : {csv_path}")
    print(f"Saved JSON : {meta_path}")
    print("\nThreshold estimates (p* @ WER = 0.10):")
    for entry in meta["threshold_estimates"]:
        p_str = f"{entry['p_star']:.5f}" if entry["p_star"] is not None else "N/A"
        print(
            f"  L={entry['L']} M={entry['M']}  delta/eps={entry['delta_eps_ratio']:.3f}"
            f"  ->  p* ~ {p_str}"
        )

    return 0


def _write_csv(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _write_meta(path: Path, meta: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())
