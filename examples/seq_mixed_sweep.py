"""
Sequential mixed-channel threshold sweep — no multiprocessing.
Runs entirely in the main process. Slow per point but starts instantly.
Automatically resumes from an existing CSV: any delta_eps whose bisection
phase is already complete (≥ bisect_steps+2 high-shot rows) is skipped.

Usage:
  python seq_mixed_sweep.py --size 12x6 --delta-eps 0.1 0.3 0.5 0.75 1.0
"""
import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from bposd import bposd_decoder
from bposd.css import css_code
from scipy.sparse import issparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qldpc import BivariateBicycleCode
from qldpc.channels import ChannelConfig, apply_channel, build_channel_probs

# ── defaults ──────────────────────────────────────────────────────────────────
SHOTS_BRACKET = 10_000   # coarse scan — fast
SHOTS_REFINE  = 50_000   # bisection — still no multiprocessing
P_START   = 0.05
P_STEP    = 0.02
P_MAX     = 0.50
BISECT    = 6
TARGET    = 0.10
OSD_ORDER = 5            # single decoder for everything
MAX_ITER  = 50
# ─────────────────────────────────────────────────────────────────────────────


def make_decoders(hx, hz, osd_order, max_iter):
    """Build bpd_x and bpd_z decoders in the calling process."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        bpd_x = bposd_decoder(
            np.asarray(hz, dtype=np.uint8),
            error_rate=0.1,
            bp_method="min_sum",
            osd_method="osd_cs",
            osd_order=osd_order,
            max_iter=max_iter,
        )
        bpd_z = bposd_decoder(
            np.asarray(hx, dtype=np.uint8),
            error_rate=0.1,
            bp_method="min_sum",
            osd_method="osd_cs",
            osd_order=osd_order,
            max_iter=max_iter,
        )
    return bpd_x, bpd_z


def run_shots(p, shots, hx, hz, lx, lz, bpd_x, bpd_z, ch_cfg, rng):
    """Run `shots` Monte Carlo trials at erasure probability p. Returns fail count."""
    N = hz.shape[1]
    fails = 0
    for _ in range(shots):
        erasure_mask = rng.random(N) < p
        x_err = np.zeros(N, dtype=np.uint8)
        z_err = np.zeros(N, dtype=np.uint8)
        apply_channel(x_err, z_err, erasure_mask, ch_cfg, rng)

        ch_probs = build_channel_probs(N, erasure_mask, ch_cfg)
        bpd_x.update_channel_probs(ch_probs)
        bpd_z.update_channel_probs(ch_probs)

        # X sector
        syn_x = (hz @ x_err) % 2
        cor_x = np.asarray(bpd_x.decode(syn_x), dtype=np.uint8)
        res_x = (x_err + cor_x) % 2
        x_fail = bool(np.any((lz @ res_x) % 2))

        # Z sector
        syn_z = (hx @ z_err) % 2
        cor_z = np.asarray(bpd_z.decode(syn_z), dtype=np.uint8)
        res_z = (z_err + cor_z) % 2
        z_fail = bool(np.any((lx @ res_z) % 2))

        if x_fail or z_fail:
            fails += 1
    return fails


def eval_p(p, shots, hx, hz, lx, lz, bpd_x, bpd_z, ch_cfg, rng, delta_eps, verbose=True):
    depolar = delta_eps * p
    ch_cfg_p = ChannelConfig(depolar_rate=depolar)
    t0 = time.time()
    fails = run_shots(p, shots, hx, hz, lx, lz, bpd_x, bpd_z, ch_cfg_p, rng)
    wer = fails / shots
    sec = time.time() - t0
    if verbose:
        print(f"    p={p:.5f}  wer={wer:.5f}  fails={fails}  shots={shots}"
              f"  depolar={depolar:.5f}  sec={sec:.1f}", flush=True)
    return wer, fails, sec


def sweep_one(L, M, delta_eps, hx, hz, lx, lz, bpd_x, bpd_z, seed,
              p_start, p_step, p_max, shots_bracket, shots_refine, bisect_steps, target):
    rng = np.random.default_rng(seed)
    ch_cfg = ChannelConfig()   # placeholder, eval_p overrides per-call

    rows = []
    channel_model = "mixed" if delta_eps > 0 else "erasure_uniform"
    N = int(2 * L * M)

    def record(p, shots, fails, sec):
        wer = fails / shots
        rows.append({
            "family": "bivariate_bicycle", "L": L, "M": M, "N": N,
            "p": round(p, 9), "shots": shots, "fails": fails, "wer": wer,
            "seconds": round(sec, 3),
            "channel_model": channel_model,
            "depolar_rate": round(delta_eps * p, 9),
            "delta_eps_ratio": delta_eps,
        })

    # ── Phase A: coarse bracket ───────────────────────────────────────────────
    p = p_start
    p_lo = p_hi = None
    wlo = whi = None
    while p <= p_max + 1e-9:
        wer, fails, sec = eval_p(p, shots_bracket, hx, hz, lx, lz,
                                  bpd_x, bpd_z, ch_cfg, rng, delta_eps)
        record(p, shots_bracket, fails, sec)
        if wer < target:
            p_lo = p
            wlo = wer
        else:
            p_hi = p
            whi = wer
            break
        p = round(p + p_step, 9)

    if p_lo is None or p_hi is None:
        print(f"  [threshold not found in range]", flush=True)
        return rows, None

    print(f"  Bracket: p_lo={p_lo:.5f}  p_hi={p_hi:.5f} — bisecting {bisect_steps}x "
          f"at {shots_refine:,} shots...", flush=True)

    # ── Phase B: weighted bisection ───────────────────────────────────────────
    # Use linear interpolation (regula falsi) to place each new point at the
    # estimated crossing, not always at the midpoint.  Falls back to midpoint
    # if the interpolated value would leave the bracket.
    # p_lo / p_hi / wlo / whi already set from Phase A — no re-evaluation needed.
    for _ in range(bisect_steps):
        # weighted next point
        if whi != wlo:
            next_p = p_lo + (target - wlo) / (whi - wlo) * (p_hi - p_lo)
        else:
            next_p = 0.5 * (p_lo + p_hi)
        # clamp inside bracket (avoid degenerate float edge cases)
        next_p = round(float(np.clip(next_p, p_lo + 1e-9, p_hi - 1e-9)), 9)
        wm, fm, sm = eval_p(next_p, shots_refine, hx, hz, lx, lz,
                             bpd_x, bpd_z, ch_cfg, rng, delta_eps)
        record(next_p, shots_refine, fm, sm)
        if wm < target:
            p_lo, wlo = next_p, wm
        else:
            p_hi, whi = next_p, wm

    # Final p* by linear interpolation on the last bracket
    if whi != wlo:
        t = (target - wlo) / (whi - wlo)
        p_star = round(float(p_lo + t * (p_hi - p_lo)), 6)
    else:
        p_star = round(0.5 * (p_lo + p_hi), 6)
    print(f"  [p* ~ {p_star}]", flush=True)
    return rows, p_star


def write_csv(path, all_rows):
    if not all_rows:
        return
    fields = ["family","L","M","N","p","shots","fails","wer","seconds",
              "channel_model","depolar_rate","delta_eps_ratio"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)


def load_existing_csv(path):
    """Load existing CSV and return rows grouped by delta_eps_ratio.

    Returns dict: {delta_eps_float: [row_dict, ...]}
    """
    if not path.exists():
        return {}
    grouped = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            de = round(float(row["delta_eps_ratio"]), 9)
            grouped.setdefault(de, []).append(row)
    return grouped


def is_complete(rows, shots_refine, bisect_steps):
    """A delta_eps sweep is complete when Phase B (bisection) finished.

    Phase B produces exactly bisect_steps + 2 rows at shots_refine
    (the two bracket endpoints + bisect_steps midpoints).
    """
    refine_rows = [r for r in rows if int(r["shots"]) == shots_refine]
    return len(refine_rows) >= bisect_steps + 2


def pstar_from_rows(rows):
    """Re-derive p* from existing bisection rows (midpoint of final bracket)."""
    refine = sorted(
        [r for r in rows if int(r["shots"]) > 10_000],
        key=lambda r: float(r["p"])
    )
    if not refine:
        return None
    target = 0.10
    lo = max((r for r in refine if float(r["wer"]) < target),
             key=lambda r: float(r["p"]), default=None)
    hi = min((r for r in refine if float(r["wer"]) >= target),
             key=lambda r: float(r["p"]), default=None)
    if lo and hi:
        return round(0.5 * (float(lo["p"]) + float(hi["p"])), 6)
    return round(float(refine[-1]["p"]), 6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size",  default="12x6",
                    help="LxM, e.g. 12x6")
    ap.add_argument("--delta-eps", nargs="+", type=float,
                    default=[0.1, 0.3, 0.5, 0.75, 1.0])
    ap.add_argument("--p-start",        type=float, default=P_START)
    ap.add_argument("--p-step",         type=float, default=P_STEP)
    ap.add_argument("--p-max",          type=float, default=P_MAX)
    ap.add_argument("--shots-bracket",  type=int,   default=SHOTS_BRACKET)
    ap.add_argument("--shots-refine",   type=int,   default=SHOTS_REFINE)
    ap.add_argument("--bisect-steps",   type=int,   default=BISECT)
    ap.add_argument("--osd-order",      type=int,   default=OSD_ORDER)
    ap.add_argument("--max-iter",       type=int,   default=MAX_ITER)
    ap.add_argument("--seed",           type=int,   default=12345)
    ap.add_argument("--out-dir",        default="results/phase1")
    args = ap.parse_args()

    L, M = (int(x) for x in args.size.lower().split("x"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"bb_{L}x{M}_seq.csv"

    print(f"Building BB code L={L} M={M}...", flush=True)
    code = BivariateBicycleCode(L=L, M=M)
    Hx_sp, Hz_sp = code.get_matrices()

    # Logical operators via css_code (only called once)
    print("Computing logical operators...", flush=True)
    hx_d = Hx_sp.toarray().astype(int) if issparse(Hx_sp) else np.asarray(Hx_sp, int)
    hz_d = Hz_sp.toarray().astype(int) if issparse(Hz_sp) else np.asarray(Hz_sp, int)
    qcode = css_code(hx=hx_d, hz=hz_d)
    lx = np.asarray(qcode.lx.toarray() if issparse(qcode.lx) else qcode.lx, dtype=np.uint8)
    lz = np.asarray(qcode.lz.toarray() if issparse(qcode.lz) else qcode.lz, dtype=np.uint8)
    hx = np.asarray(qcode.hx.toarray() if issparse(qcode.hx) else qcode.hx, dtype=np.uint8)
    hz = np.asarray(qcode.hz.toarray() if issparse(qcode.hz) else qcode.hz, dtype=np.uint8)
    K  = int(getattr(qcode, "K", lx.shape[0]))
    print(f"Code [[{2*L*M},{K}]]  Hx={hx.shape}  Hz={hz.shape}", flush=True)

    # Build decoders ONCE
    print(f"Building OSD-{args.osd_order} decoders...", flush=True)
    bpd_x, bpd_z = make_decoders(hx, hz, args.osd_order, args.max_iter)
    print("Decoders ready. Starting sweep.\n", flush=True)

    # ── Resume: load any existing results ────────────────────────────────────
    existing = load_existing_csv(csv_path)
    if existing:
        completed = [de for de, rows in existing.items()
                     if is_complete(rows, args.shots_refine, args.bisect_steps)]
        print(f"Existing CSV found: {csv_path.name}", flush=True)
        print(f"  Already complete: delta_eps = {sorted(completed)}", flush=True)
    else:
        completed = []

    # Seed all_rows with previously completed data so the CSV stays intact
    all_rows = []
    for de_rows in existing.values():
        all_rows.extend(
            {k: (float(v) if k in ("p","wer","seconds","depolar_rate","delta_eps_ratio")
                 else int(v) if k in ("L","M","N","shots","fails")
                 else v)
             for k, v in row.items()}
            for row in de_rows
        )

    results = {}

    for de in args.delta_eps:
        de_key = round(de, 9)
        print(f"{'='*60}", flush=True)
        # ── Skip if already complete ──────────────────────────────────────
        if de_key in completed:
            p_star = pstar_from_rows(existing[de_key])
            results[de] = p_star
            print(f"delta/eps = {de:.3f}  [SKIPPED — already in {csv_path.name}]"
                  f"  p* ~ {p_star}", flush=True)
            continue

        print(f"delta/eps = {de:.3f}  (depolar = {de} * p per point)", flush=True)
        rows, p_star = sweep_one(
            L, M, de, hx, hz, lx, lz, bpd_x, bpd_z, args.seed,
            args.p_start, args.p_step, args.p_max,
            args.shots_bracket, args.shots_refine,
            args.bisect_steps, TARGET,
        )
        all_rows.extend(rows)
        results[de] = p_star
        write_csv(csv_path, all_rows)
        print(f"  [Saved -> {csv_path.name}]", flush=True)

    print(f"\n{'='*60}")
    print(f"Saved: {csv_path}")
    print("Threshold estimates (p* @ WER=0.10):")
    for de, ps in results.items():
        print(f"  delta/eps={de:.3f}  ->  p* ~ {ps if ps else 'N/A'}")


if __name__ == "__main__":
    main()
