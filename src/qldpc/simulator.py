"""
Quantum Error Correction Simulator

High-performance Monte Carlo simulation for quantum error correction codes
with parallel processing support.
"""

import numpy as np
from numpy.random import SeedSequence
from scipy.sparse import issparse
import multiprocessing
import time
from typing import Dict, List, Tuple, Optional, Any

from .code import BivariateBicycleCode
from .decoder import DecoderConfig, create_decoders


def _coerce_correction(correction: Any, n: int, dtype: np.dtype) -> np.ndarray:
    if not isinstance(correction, np.ndarray):
        correction = np.array(correction, dtype=dtype)
    else:
        correction = correction.astype(dtype, copy=False)
    if correction.shape != (n,):
        correction = correction.flatten()[:n].astype(dtype, copy=False)
    return correction


def worker_simulation(args: Tuple) -> int:
    """
    Runs a batch of Monte Carlo simulations on a single core.
    
    This function is designed to be called by multiprocessing.Pool.
    It instantiates the decoder once per worker, then runs multiple
    simulation shots with updated channel probabilities.
    
    Parameters
    ----------
    args : tuple
        (seed, shots, erasure_p, hx_csr, hz_csr, config)
        
    Returns
    -------
    int
        Number of logical errors encountered
    """
    seed, shots, erasure_p, hx_csr, hz_csr, config = args
    rng = np.random.default_rng(seed)
    
    # 1. Setup Code & Decoders (Expensive, do once per worker)
    qcode, bpd_x, bpd_z = create_decoders(hx_csr, hz_csr, config, initial_error_rate=erasure_p)
    
    logical_fails = 0
    background_error = 1e-10  # Assumed essentially perfect if not erased
    
    # 2. The Loop (Fast)
    for _ in range(shots):
        # A. Generate Erasures (Hardware Layer)
        # Erasure mask: 1 if erased, 0 if clean
        erasure_mask = rng.random(qcode.N) < erasure_p
        
        # B. Apply Noise
        # Pure erasure channel: if erased, randomized Pauli X/Z.
        # For a maximally mixed erasure, X and Z components are independent Bernoulli(0.5).
        x_error = np.zeros(qcode.N, dtype=np.uint8)
        z_error = np.zeros(qcode.N, dtype=np.uint8)
        
        # Apply random bit flips ONLY on erased qubits (Maximal Entropy)
        # In erasure conversion, 50% chance of X/Z error on erased qubit
        rand_x = rng.integers(0, 2, size=qcode.N, dtype=np.uint8)
        rand_z = rng.integers(0, 2, size=qcode.N, dtype=np.uint8)
        x_error[erasure_mask] = rand_x[erasure_mask]
        z_error[erasure_mask] = rand_z[erasure_mask]
        
        # C. Hardware Flagging (The "Secret Weapon")
        # We tell the decoder which bits are erased by setting their prob to 0.5 (LLR = 0)
        # Normal bits get a very low error probability (simulating high fidelity background)
        channel_probs = np.full(qcode.N, background_error)
        channel_probs[erasure_mask] = 0.5
        
        # Update decoder priors
        bpd_x.update_channel_probs(channel_probs)
        bpd_z.update_channel_probs(channel_probs)
        
        # D. Decode X errors using Hz checks
        x_syndrome = qcode.hz @ x_error % 2
        x_correction = _coerce_correction(bpd_x.decode(x_syndrome), qcode.N, np.uint8)
        x_fail = False
        if not np.array_equal((qcode.hz @ x_correction) % 2, x_syndrome):
            x_fail = True
        else:
            x_residual = (x_error + x_correction) % 2
            if not np.all((qcode.hz @ x_residual) % 2 == 0):
                x_fail = True
            elif np.any((qcode.lz @ x_residual) % 2):
                x_fail = True

        # E. Decode Z errors using Hx checks
        z_syndrome = qcode.hx @ z_error % 2
        z_correction = _coerce_correction(bpd_z.decode(z_syndrome), qcode.N, np.uint8)
        z_fail = False
        if not np.array_equal((qcode.hx @ z_correction) % 2, z_syndrome):
            z_fail = True
        else:
            z_residual = (z_error + z_correction) % 2
            if not np.all((qcode.hx @ z_residual) % 2 == 0):
                z_fail = True
            elif np.any((qcode.lx @ z_residual) % 2):
                z_fail = True

        # Word error if either X or Z sector fails
        if x_fail or z_fail:
            logical_fails += 1

    return logical_fails


# ==========================
# Prepared (persistent) worker path
#   - compute qcode/logicals once in parent
#   - initialize decoder once per worker
#   - reuse for many p evaluations
# ==========================
_PREPARED = {}


def _prepared_worker_init(
    hx: np.ndarray,
    hz: np.ndarray,
    lx: np.ndarray,
    lz: np.ndarray,
    config: DecoderConfig,
    background_error: float,
):
    """
    Initializer for multiprocessing workers.
    Stores Hz/Lz and instantiates a BP-OSD decoder once per worker.
    """
    # Local import to avoid importing bposd in parent unnecessarily
    from bposd import bposd_decoder
    import warnings

    hx_u8 = np.asarray(hx, dtype=np.uint8)
    hz_u8 = np.asarray(hz, dtype=np.uint8)
    lx_u8 = np.asarray(lx, dtype=np.uint8)
    lz_u8 = np.asarray(lz, dtype=np.uint8)
    n = int(hz_u8.shape[1])

    # Store in module globals (per process)
    _PREPARED["hx"] = hx_u8
    _PREPARED["hz"] = hz_u8
    _PREPARED["lx"] = lx_u8
    _PREPARED["lz"] = lz_u8
    _PREPARED["N"] = n
    _PREPARED["background_error"] = float(background_error)

    # Decoder initialized once; error_rate here is just an initial placeholder.
    # Suppress deprecation warning for legacy bposd_decoder API (still works)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*old syntax for the `bposd_decoder`.*",
        )
        _PREPARED["bpd_x"] = bposd_decoder(
            hz_u8,
            error_rate=0.1,
            bp_method=config.bp_method,
            osd_method=config.osd_method,
            osd_order=config.osd_order,
            max_iter=config.max_iter,
        )
        _PREPARED["bpd_z"] = bposd_decoder(
            hx_u8,
            error_rate=0.1,
            bp_method=config.bp_method,
            osd_method=config.osd_method,
            osd_order=config.osd_order,
            max_iter=config.max_iter,
        )


def _prepared_worker_simulation(args: Tuple) -> int:
    """
    Worker simulation using pre-initialized decoder and precomputed matrices.
    Args: (seed, shots, erasure_p)
    """
    seed, shots, erasure_p = args
    rng = np.random.default_rng(seed)

    hx = _PREPARED["hx"]
    hz = _PREPARED["hz"]
    lx = _PREPARED["lx"]
    lz = _PREPARED["lz"]
    N = _PREPARED["N"]
    background_error = _PREPARED["background_error"]
    bpd_x = _PREPARED["bpd_x"]
    bpd_z = _PREPARED["bpd_z"]

    logical_fails = 0

    for _ in range(shots):
        erasure_mask = rng.random(N) < erasure_p

        x_error = np.zeros(N, dtype=np.uint8)
        z_error = np.zeros(N, dtype=np.uint8)
        rand_x = rng.integers(0, 2, size=N, dtype=np.uint8)
        rand_z = rng.integers(0, 2, size=N, dtype=np.uint8)
        x_error[erasure_mask] = rand_x[erasure_mask]
        z_error[erasure_mask] = rand_z[erasure_mask]

        channel_probs = np.full(N, background_error, dtype=float)
        channel_probs[erasure_mask] = 0.5
        bpd_x.update_channel_probs(channel_probs)
        bpd_z.update_channel_probs(channel_probs)

        # X errors
        x_syndrome = (hz @ x_error) % 2
        x_correction = _coerce_correction(bpd_x.decode(x_syndrome), N, np.uint8)
        x_fail = False
        if not np.array_equal((hz @ x_correction) % 2, x_syndrome):
            x_fail = True
        else:
            x_residual = (x_error + x_correction) % 2
            if not np.all((hz @ x_residual) % 2 == 0):
                x_fail = True
            elif np.any((lz @ x_residual) % 2):
                x_fail = True

        # Z errors
        z_syndrome = (hx @ z_error) % 2
        z_correction = _coerce_correction(bpd_z.decode(z_syndrome), N, np.uint8)
        z_fail = False
        if not np.array_equal((hx @ z_correction) % 2, z_syndrome):
            z_fail = True
        else:
            z_residual = (z_error + z_correction) % 2
            if not np.all((hx @ z_residual) % 2 == 0):
                z_fail = True
            elif np.any((lx @ z_residual) % 2):
                z_fail = True

        if x_fail or z_fail:
            logical_fails += 1

    return logical_fails


class QuantumSimulator:
    """
    High-performance quantum error correction simulator.
    
    This class orchestrates parallel Monte Carlo simulations to determine
    the error threshold of quantum error correction codes.
    
    Parameters
    ----------
    code : BivariateBicycleCode
        The quantum code to simulate
    config : DecoderConfig, optional
        Decoder configuration. If None, uses default settings.
    num_cores : int, optional
        Number of CPU cores to use. If None, uses all but one.
    """
    
    def __init__(
        self,
        code: BivariateBicycleCode,
        config: DecoderConfig = None,
        num_cores: int = None,
        seed: Optional[int] = None,
    ):
        self.code = code
        self.config = config or DecoderConfig()
        self.num_cores = num_cores or max(1, multiprocessing.cpu_count() - 1)
        self.seed = seed
        
        # Pre-compute matrices
        self.Hx, self.Hz = code.get_matrices()
        
    def run_experiment(
        self,
        erasure_rates: List[float],
        total_shots: int = 5000,
        verbose: bool = True,
        return_details: bool = False,
        return_seeds: bool = False,
        seed: Optional[int] = None,
        pool=None,
        close_pool: bool = True,
    ):
        """
        Run Monte Carlo simulation across multiple erasure rates.
        
        Parameters
        ----------
        erasure_rates : list of float
            List of erasure probabilities to test
        total_shots : int, default=5000
            Total number of simulation shots per erasure rate
        verbose : bool, default=True
            Whether to print progress information
            
        Returns
        -------
        dict
            If return_details=False: {p: wer}
            If return_details=True: {p: {"wer": float, "fails": int, "shots": int, "seconds": float}}

        Notes
        -----
        WER is defined as a logical failure in either the X or Z sector for an erasure-aware decoder.
        """
        if verbose:
            code_name = self.code.__class__.__name__
            code_params = []
            if hasattr(self.code, "L"):
                code_params.append(f"L={getattr(self.code, 'L')}")
            if hasattr(self.code, "M"):
                code_params.append(f"M={getattr(self.code, 'M')}")
            code_params.append(f"N={getattr(self.code, 'N', 'unknown')}")
            print(f"--- CONSTRUCTING {code_name} ({', '.join(code_params)}) ---")
            print(f"Matrix Shapes: Hx {self.Hx.shape}, Hz {self.Hz.shape}")
            print("Code constructed. Starting Simulation...")
            print(f"{'Erasure Rate':<15} | {'Shots':<10} | {'Log Errors':<10} | {'WER':<10} | {'Time (s)':<10}")
            print("-" * 70)

        created_pool = False
        if pool is None:
            pool = multiprocessing.Pool(self.num_cores)
            created_pool = True
        results = {}
        base_seed = self.seed if seed is None else seed

        for p in erasure_rates:
            # Distribute work across cores
            shot_counts = self._split_shots(total_shots, self.num_cores)
            worker_seeds = self._make_worker_seeds(p, self.num_cores, base_seed)
            args = []
            used_worker_seeds = []
            for i in range(self.num_cores):
                if shot_counts[i] <= 0:
                    continue
                args.append((worker_seeds[i], shot_counts[i], p, self.Hx, self.Hz, self.config))
                used_worker_seeds.append(worker_seeds[i])
            
            start_time = time.time()
            
            # Run Parallel
            worker_results = pool.map(worker_simulation, args)
            
            # Aggregate
            total_fails = sum(worker_results)
            actual_shots = sum(shot_counts)
            wer = total_fails / actual_shots
            elapsed = time.time() - start_time
            
            if verbose:
                print(f"{p:<15.4f} | {actual_shots:<10} | {total_fails:<10} | "
                      f"{wer:<10.5f} | {elapsed:<10.2f}")
            
            if return_details:
                results[p] = {
                    "wer": float(wer),
                    "fails": int(total_fails),
                    "shots": int(actual_shots),
                    "seconds": float(elapsed),
                }
                if return_seeds:
                    results[p]["worker_seeds"] = list(used_worker_seeds)
            else:
                results[p] = float(wer)

        if (created_pool and close_pool) or (pool is not None and close_pool):
            pool.close()
            pool.join()
        
        if verbose:
            print("\n--- SIMULATION COMPLETE ---")
            print("Interpretation: Look for the 'Break-even' point.")
            print("Note: WER counts logical failure in either X or Z sector for erasure-aware decoding.")
        
        return results

    def run_point(
        self,
        erasure_rate: float,
        total_shots: int,
        pool=None,
        seed: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Run a single erasure rate point and return detailed stats.
        """
        created_pool = False
        if pool is None:
            pool = multiprocessing.Pool(self.num_cores)
            created_pool = True

        base_seed = self.seed if seed is None else seed
        shot_counts = self._split_shots(total_shots, self.num_cores)
        worker_seeds = self._make_worker_seeds(erasure_rate, self.num_cores, base_seed)
        args = []
        used_worker_seeds = []
        for i in range(self.num_cores):
            if shot_counts[i] <= 0:
                continue
            args.append((worker_seeds[i], shot_counts[i], erasure_rate, self.Hx, self.Hz, self.config))
            used_worker_seeds.append(worker_seeds[i])
        start_time = time.time()
        worker_results = pool.map(worker_simulation, args)

        total_fails = int(sum(worker_results))
        actual_shots = int(sum(shot_counts))
        wer = float(total_fails / actual_shots)
        elapsed = float(time.time() - start_time)

        if created_pool:
            pool.close()
            pool.join()

        return {
            "wer": wer,
            "fails": total_fails,
            "shots": actual_shots,
            "seconds": elapsed,
            "worker_seeds": list(used_worker_seeds),
        }

    def make_prepared_pool(self, background_error: float = 1e-10):
        """
        Create a multiprocessing pool whose workers are initialized once with:
        - dense Hz and dense Lz (uint8)
        - a BP-OSD decoder instance

        Returns: (pool, info_dict)
        """
        qcode, _, _ = create_decoders(self.Hx, self.Hz, self.config, initial_error_rate=0.1)

        hx_src = qcode.hx
        hz_src = qcode.hz
        lx_src = qcode.lx
        lz_src = qcode.lz
        # bposd may store these as sparse matrices for some sizes/configs
        if issparse(hx_src):
            hx_src = hx_src.toarray()
        if issparse(hz_src):
            hz_src = hz_src.toarray()
        if issparse(lx_src):
            lx_src = lx_src.toarray()
        if issparse(lz_src):
            lz_src = lz_src.toarray()

        hx = np.asarray(hx_src, dtype=np.uint8)
        hz = np.asarray(hz_src, dtype=np.uint8)
        lx = np.asarray(lx_src, dtype=np.uint8)
        lz = np.asarray(lz_src, dtype=np.uint8)
        K = int(getattr(qcode, "K", lz.shape[0] if hasattr(lz, "shape") else 0))

        pool = multiprocessing.Pool(
            self.num_cores,
            initializer=_prepared_worker_init,
            initargs=(hx, hz, lx, lz, self.config, float(background_error)),
        )
        info = {
            "N": int(hz.shape[1]),
            "M": int(hz.shape[0]),
            "K": K,
            "hx_shape": tuple(hx.shape),
            "hz_shape": tuple(hz.shape),
            "lx_shape": tuple(lx.shape),
            "lz_shape": tuple(lz.shape),
        }
        if hasattr(self.code, "L"):
            info["L"] = int(getattr(self.code, "L"))
        if hasattr(self.code, "M"):
            info["M_param"] = int(getattr(self.code, "M"))
        return pool, info

    def run_point_prepared(
        self,
        erasure_rate: float,
        total_shots: int,
        pool,
        seed: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Run a single p using a prepared pool (persistent workers).
        """
        base_seed = self.seed if seed is None else seed
        shot_counts = self._split_shots(total_shots, self.num_cores)
        worker_seeds = self._make_worker_seeds(erasure_rate, self.num_cores, base_seed)
        args = []
        used_worker_seeds = []
        for i in range(self.num_cores):
            if shot_counts[i] <= 0:
                continue
            args.append((worker_seeds[i], shot_counts[i], float(erasure_rate)))
            used_worker_seeds.append(worker_seeds[i])
        start_time = time.time()
        worker_results = pool.map(_prepared_worker_simulation, args)
        total_fails = int(sum(worker_results))
        actual_shots = int(sum(shot_counts))
        wer = float(total_fails / actual_shots)
        elapsed = float(time.time() - start_time)
        return {
            "wer": wer,
            "fails": total_fails,
            "shots": actual_shots,
            "seconds": elapsed,
            "worker_seeds": list(used_worker_seeds),
        }

    @staticmethod
    def _split_shots(total_shots: int, num_workers: int) -> List[int]:
        if num_workers <= 0:
            return []
        base = total_shots // num_workers
        remainder = total_shots % num_workers
        return [base + (1 if i < remainder else 0) for i in range(num_workers)]

    def _make_worker_seeds(self, erasure_rate: float, num_workers: int, base_seed: Optional[int]) -> List[int]:
        if num_workers <= 0:
            return []
        if base_seed is None:
            ss = SeedSequence()
        else:
            key_parts = [int(base_seed)]
            for attr in ("L", "M", "N"):
                if hasattr(self.code, attr):
                    key_parts.append(int(getattr(self.code, attr)))
            p_key = int(round(float(erasure_rate) * 1_000_000_000))
            key_parts.append(p_key)
            ss = SeedSequence(key_parts)
        children = ss.spawn(num_workers)
        return [int(c.generate_state(1, dtype=np.uint32)[0]) for c in children]

