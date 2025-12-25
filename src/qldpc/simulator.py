"""
Quantum Error Correction Simulator

High-performance Monte Carlo simulation for quantum error correction codes
with parallel processing support.
"""

import numpy as np
from scipy.sparse import csr_matrix, issparse
import multiprocessing
import time
from typing import Dict, List, Tuple, Optional, Any

from .code import BivariateBicycleCode
from .decoder import DecoderConfig, create_decoder


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
    np.random.seed(seed)
    
    # 1. Setup Code & Decoder (Expensive, do once per worker)
    qcode, bpd = create_decoder(hx_csr, hz_csr, config, initial_error_rate=erasure_p)
    
    logical_fails = 0
    background_error = 1e-10  # Assumed essentially perfect if not erased
    
    # 2. The Loop (Fast)
    for _ in range(shots):
        # A. Generate Erasures (Hardware Layer)
        # Erasure mask: 1 if erased, 0 if clean
        erasure_mask = np.random.random(qcode.N) < erasure_p
        
        # B. Apply Noise
        # Pure erasure channel: if erased, randomized Pauli X/Z.
        # Here we only track X errors for the Z-checks (CSS codes decouple X/Z)
        error = np.zeros(qcode.N, dtype=int)
        
        # Apply random bit flips ONLY on erased qubits (Maximal Entropy)
        # In erasure conversion, 50% chance of X error on erased qubit
        random_flips = np.random.randint(0, 2, size=qcode.N)
        error[erasure_mask] = random_flips[erasure_mask]
        
        # C. Hardware Flagging (The "Secret Weapon")
        # We tell the decoder which bits are erased by setting their prob to 0.5 (LLR = 0)
        # Normal bits get a very low error probability (simulating high fidelity background)
        channel_probs = np.full(qcode.N, background_error)
        channel_probs[erasure_mask] = 0.5
        
        # Update decoder priors
        bpd.update_channel_probs(channel_probs)
        
        # D. Decode
        syndrome = qcode.hz @ error % 2
        correction = bpd.decode(syndrome)
        
        # Ensure correction is a numpy array of the correct shape
        if not isinstance(correction, np.ndarray):
            correction = np.array(correction, dtype=int)
        if correction.shape != (qcode.N,):
            # Handle case where correction might be 1D but wrong length, or 2D
            if correction.ndim == 2 and correction.shape[1] == qcode.N:
                correction = correction[0]
            elif correction.ndim == 1 and len(correction) != qcode.N:
                # Pad or truncate if needed
                new_correction = np.zeros(qcode.N, dtype=int)
                min_len = min(len(correction), qcode.N)
                new_correction[:min_len] = correction[:min_len]
                correction = new_correction
            else:
                correction = correction.flatten()[:qcode.N]
        
        # E. Validate (Check Logic)
        # First verify the correction actually fixes the syndrome
        correction_syndrome = qcode.hz @ correction % 2
        if not np.array_equal(correction_syndrome, syndrome):
            # Decoder failed to find a valid correction
            logical_fails += 1
            continue
        
        residual = (error + correction) % 2
        
        # Success condition: Residual must be a stabilizer (syndrome 0) 
        # AND it must NOT trigger a logical operator.
        
        # Check 1: Did syndrome converge? (residual should have zero syndrome)
        residual_syndrome = qcode.hz @ residual % 2
        if not np.all(residual_syndrome == 0):
            logical_fails += 1
            continue

        # Check 2: Logical Error Check
        # We check if the residual has non-trivial overlap with Logical Z operators
        # In CSS X-basis decoding, X errors can flip logical Z operators
        # (qcode.lx contains the Logical X operators, qcode.lz contains Logical Z)
        
        # If residual has odd overlap with any Logical Z, we have a logical error
        logical_overlap = qcode.lz @ residual % 2
        if np.any(logical_overlap):
            logical_fails += 1

    return logical_fails


# ==========================
# Prepared (persistent) worker path
#   - compute qcode/logicals once in parent
#   - initialize decoder once per worker
#   - reuse for many p evaluations
# ==========================
_PREPARED = {}


def _prepared_worker_init(hz: np.ndarray, lz: np.ndarray, config: DecoderConfig, background_error: float):
    """
    Initializer for multiprocessing workers.
    Stores Hz/Lz and instantiates a BP-OSD decoder once per worker.
    """
    # Local import to avoid importing bposd in parent unnecessarily
    from bposd import bposd_decoder
    import warnings

    hz_u8 = np.asarray(hz, dtype=np.uint8)
    lz_u8 = np.asarray(lz, dtype=np.uint8)
    n = int(hz_u8.shape[1])

    # Store in module globals (per process)
    _PREPARED["hz"] = hz_u8
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
        _PREPARED["bpd"] = bposd_decoder(
            hz_u8,
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
    np.random.seed(seed)

    hz = _PREPARED["hz"]
    lz = _PREPARED["lz"]
    N = _PREPARED["N"]
    background_error = _PREPARED["background_error"]
    bpd = _PREPARED["bpd"]

    logical_fails = 0

    for _ in range(shots):
        erasure_mask = np.random.random(N) < erasure_p

        error = np.zeros(N, dtype=np.uint8)
        random_flips = np.random.randint(0, 2, size=N, dtype=np.uint8)
        error[erasure_mask] = random_flips[erasure_mask]

        channel_probs = np.full(N, background_error, dtype=float)
        channel_probs[erasure_mask] = 0.5
        bpd.update_channel_probs(channel_probs)

        syndrome = (hz @ error) % 2
        correction = bpd.decode(syndrome)
        if not isinstance(correction, np.ndarray):
            correction = np.array(correction, dtype=np.uint8)
        else:
            correction = correction.astype(np.uint8, copy=False)
        if correction.shape != (N,):
            correction = correction.flatten()[:N].astype(np.uint8, copy=False)

        # Validate correction fixes syndrome
        if not np.array_equal((hz @ correction) % 2, syndrome):
            logical_fails += 1
            continue

        residual = (error + correction) % 2
        if not np.all((hz @ residual) % 2 == 0):
            logical_fails += 1
            continue

        if np.any((lz @ residual) % 2):
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
    
    def __init__(self, code: BivariateBicycleCode, 
                 config: DecoderConfig = None,
                 num_cores: int = None):
        self.code = code
        self.config = config or DecoderConfig()
        self.num_cores = num_cores or max(1, multiprocessing.cpu_count() - 1)
        
        # Pre-compute matrices
        self.Hx, self.Hz = code.get_matrices()
        
    def run_experiment(
        self,
        erasure_rates: List[float],
        total_shots: int = 5000,
        verbose: bool = True,
        return_details: bool = False,
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
        """
        if verbose:
            print("--- CONSTRUCTING [[144, 12, 12]] BIVARIATE BICYCLE CODE ---")
            print(f"Matrix Shapes: Hx {self.Hx.shape}, Hz {self.Hz.shape}")
            print("Code constructed. Starting Simulation...")
            print(f"{'Erasure Rate':<15} | {'Shots':<10} | {'Log Errors':<10} | {'WER':<10} | {'Time (s)':<10}")
            print("-" * 70)

        created_pool = False
        if pool is None:
            pool = multiprocessing.Pool(self.num_cores)
            created_pool = True
        results = {}

        for p in erasure_rates:
            # Distribute work across cores
            shots_per_worker = total_shots // self.num_cores
            args = [
                (np.random.randint(1e9), shots_per_worker, p, 
                 self.Hx, self.Hz, self.config) 
                for _ in range(self.num_cores)
            ]
            
            start_time = time.time()
            
            # Run Parallel
            worker_results = pool.map(worker_simulation, args)
            
            # Aggregate
            total_fails = sum(worker_results)
            actual_shots = shots_per_worker * self.num_cores
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
            else:
                results[p] = float(wer)

        if (created_pool and close_pool) or (pool is not None and close_pool):
            pool.close()
            pool.join()
        
        if verbose:
            print("\n--- SIMULATION COMPLETE ---")
            print("Interpretation: Look for the 'Break-even' point.")
            print("For a standard Surface Code (d=12), the threshold is usually around 10-15% for pure erasure.")
            print("Since this code has k=12 (encodes 12 qubits), a WER < 0.1 at p=0.08 is spectacular.")
        
        return results

    def run_point(self, erasure_rate: float, total_shots: int, pool=None) -> Dict[str, float]:
        """
        Run a single erasure rate point and return detailed stats.
        """
        created_pool = False
        if pool is None:
            pool = multiprocessing.Pool(self.num_cores)
            created_pool = True

        shots_per_worker = total_shots // self.num_cores
        args = [
            (np.random.randint(1e9), shots_per_worker, erasure_rate, self.Hx, self.Hz, self.config)
            for _ in range(self.num_cores)
        ]
        start_time = time.time()
        worker_results = pool.map(worker_simulation, args)

        total_fails = int(sum(worker_results))
        actual_shots = int(shots_per_worker * self.num_cores)
        wer = float(total_fails / actual_shots)
        elapsed = float(time.time() - start_time)

        if created_pool:
            pool.close()
            pool.join()

        return {"wer": wer, "fails": total_fails, "shots": actual_shots, "seconds": elapsed}

    def make_prepared_pool(self, background_error: float = 1e-10):
        """
        Create a multiprocessing pool whose workers are initialized once with:
        - dense Hz and dense Lz (uint8)
        - a BP-OSD decoder instance

        Returns: (pool, info_dict)
        """
        qcode, _ = create_decoder(self.Hx, self.Hz, self.config, initial_error_rate=0.1)

        hz_src = qcode.hz
        lz_src = qcode.lz
        # bposd may store these as sparse matrices for some sizes/configs
        if issparse(hz_src):
            hz_src = hz_src.toarray()
        if issparse(lz_src):
            lz_src = lz_src.toarray()

        hz = np.asarray(hz_src, dtype=np.uint8)
        lz = np.asarray(lz_src, dtype=np.uint8)
        K = int(getattr(qcode, "K", lz.shape[0] if hasattr(lz, "shape") else 0))

        pool = multiprocessing.Pool(
            self.num_cores,
            initializer=_prepared_worker_init,
            initargs=(hz, lz, self.config, float(background_error)),
        )
        info = {
            "N": int(hz.shape[1]),
            "M": int(hz.shape[0]),
            "K": K,
            "hz_shape": tuple(hz.shape),
            "lz_shape": tuple(lz.shape),
        }
        return pool, info

    def run_point_prepared(self, erasure_rate: float, total_shots: int, pool) -> Dict[str, float]:
        """
        Run a single p using a prepared pool (persistent workers).
        """
        shots_per_worker = total_shots // self.num_cores
        args = [(np.random.randint(1_000_000_000), shots_per_worker, float(erasure_rate)) for _ in range(self.num_cores)]
        start_time = time.time()
        worker_results = pool.map(_prepared_worker_simulation, args)
        total_fails = int(sum(worker_results))
        actual_shots = int(shots_per_worker * self.num_cores)
        wer = float(total_fails / actual_shots)
        elapsed = float(time.time() - start_time)
        return {"wer": wer, "fails": total_fails, "shots": actual_shots, "seconds": elapsed}

