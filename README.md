# QLDPC: Quantum Low-Density Parity-Check Code Simulation Framework

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A high-performance simulation framework for quantum error correction codes, specifically optimized for **Bivariate Bicycle Codes** and erasure channel analysis. This implementation focuses on code capacity simulations using matrix-based methods and probability calculations, providing a mathematically rigorous way to determine error thresholds without simulating the physics of every gate.

## 🎯 Features

- **Fast Code Capacity Simulation**: Matrix + Probability only approach (no heavy circuit compilation)
- **Memory Optimized**: Uses `scipy.sparse.csr_matrix` for efficient memory usage (<100MB RAM for typical code sizes)
- **Parallel Processing**: Multiprocessing support to saturate all CPU cores
- **Hot-Path Optimization**: Decoder instantiated once per core, only priors updated per shot
- **Parameterized Codes**: Bivariate Bicycle Codes with arbitrary (L, M) parameters
- **Example: Gross Code**: The famous [[144, 12, 12]] code (L=12, M=6) with remarkable threshold properties
- **Scaling Studies**: Built-in tools for sweeping code sizes and finding error thresholds

## 📋 Prerequisites

- Python 3.7 or higher
- Required packages (see `requirements.txt`):
  - `numpy >= 1.21.0`
  - `scipy >= 1.7.0`
  - `ldpc >= 0.1.0`
  - `bposd >= 0.1.0`

## 🚀 Installation

### Option 1: Install from source

```bash
# Clone the repository
git clone https://github.com/yourusername/QLDPC.git
cd QLDPC

# Install dependencies
pip install -r requirements.txt

# Install package (optional, for development)
pip install -e .
```

### Option 2: Direct installation of dependencies

```bash
pip install numpy scipy ldpc bposd
```

## 📖 Usage

### Quick Start

Run the example simulation:

```bash
python examples/run_simulation.py
```

### Programmatic Usage

```python
from qldpc import BivariateBicycleCode, QuantumSimulator, DecoderConfig

# Create the code
code = BivariateBicycleCode(L=12, M=6)

# Configure decoder (optional)
config = DecoderConfig(
    bp_method="min_sum",
    osd_method="osd_cs",
    osd_order=10,
    max_iter=50
)

# Create simulator
simulator = QuantumSimulator(code, config=config)

# Run experiment
erasure_rates = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
results = simulator.run_experiment(
    erasure_rates=erasure_rates,
    total_shots=5000,
    verbose=True
)
```

## 📊 Understanding the Results

The simulation outputs a table showing:

- **Erasure Rate**: The probability of qubit erasure
- **Shots**: Number of Monte Carlo samples
- **Log Errors**: Number of logical errors detected
- **WER**: Word Error Rate (logical errors / total shots)

### Interpretation

- **Low Error (p=0.02)**: WER should be extremely close to 0, confirming the code works
- **Threshold**: As you approach p=0.10 or 0.12, the error rate will spike
- **Success Criteria**: If you get extremely low error rates at p=0.06 (6% erasure), you have successfully implemented state-of-the-art QEC

For comparison:
- A standard Surface Code (d=12) typically has a threshold around 10-15% for pure erasure
- A standard memory without error correction would fail 100% of the time at 6% erasure over 144 qubits
- A WER < 0.1 at p=0.08 is spectacular for a code encoding 12 qubits

## 🏆 Performance Benchmarks

### Threshold Results (WER = 0.10)

| Code | N | K | Rate | p* (threshold) | vs Surface Code |
|------|---|---|------|----------------|-----------------|
| BB 12×6 | 144 | 12 | 0.083 | ~0.383 | **2.3× better** |
| BB 18×9 | 324 | 18 | 0.056 | ~0.426 | **2.5× better** |
| BB 24×12 | 576 | 24 | 0.042 | ~0.454 | **2.6× better** |
| BB 30×15 | 900 | 8 | 0.009 | ~0.471 | **2.7× better** |
| BB 36×18 | 1296 | 12 | 0.009 | ~0.473 | **2.7× better** |

*Surface code baseline: p* ≈ 0.16-0.18 (uninformed MWPM decoding)*

### Runtime Performance

On a **16-core AMD/Intel CPU** with **50,000 shots**:

| Code Size | N | Time per p-point | Memory |
|-----------|---|------------------|--------|
| 12×6 | 144 | ~5-10 sec | ~50 MB |
| 36×18 | 1296 | ~180-300 sec | ~280 MB |

See `docs/performance.md` for detailed profiling information.

## 📈 Code Comparison

### Bivariate Bicycle vs Other QLDPC Codes

| Property | Bivariate Bicycle | Surface Code | Hypergraph Product | Quantum Expander |
|----------|-------------------|--------------|-------------------|------------------|
| **Rate** | Low-Medium (0.01-0.1) | Low (~0.01) | Medium (~0.1) | Medium-High |
| **Threshold (erasure)** | **~40-47%** | ~16-18% | ~30-40% | Variable |
| **Distance** | Good (√N scaling) | Excellent (√N) | Good | Variable |
| **Construction** | Algebraic (simple) | Geometric | Algebraic | Probabilistic |
| **Decoding** | BP-OSD | MWPM (fast) | BP-OSD | BP-based |
| **Implementation** | ✅ This repo | ✅ PyMatching | Research | Research |

### When to Use Each Code

- **Bivariate Bicycle**: High erasure tolerance, moderate rate, algebraic construction
- **Surface Code**: Best-studied, fast decoding, topological protection, low rate
- **Hypergraph Product**: Balanced rate/distance, flexible construction
- **Quantum Expander**: High-rate applications, experimental

## 🏗️ Project Structure

```
QLDPC/
├── src/
│   └── qldpc/
│       ├── __init__.py
│       ├── code.py          # Code construction (BivariateBicycleCode, ToricCode)
│       ├── decoder.py       # Decoder configuration (BP-OSD)
│       ├── simulator.py     # Monte Carlo simulation engine
│       └── toric_code.py    # Toric surface code baseline
├── examples/
│   ├── run_simulation.py    # Quick start example
│   ├── sweep_scaling.py     # Multi-size threshold sweeps
│   ├── sweep_surface_baseline.py  # Surface code comparison
│   ├── find_valid_sizes.py  # Find (L,M) with K>0
│   ├── plot_results.py      # Generate publication-quality plots
│   └── compare_runs.py      # Compare different configurations
├── docs/
│   └── performance.md       # Detailed performance guide
├── tests/
│   ├── test_code.py         # Unit tests
│   └── ...                  # More tests
├── results/                 # Output directory for CSV/JSON/plots
├── requirements.txt         # Python dependencies
├── setup.py                 # Package setup
├── pyproject.toml          # Modern Python packaging
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## 🔬 Technical Details

### Code Construction

The Bivariate Bicycle Code is constructed using:
- **Polynomials**: A = x³ + y + y², B = y³ + x + x²
- **Lifted Product**: Hx = [A | B], Hz = [Bᵀ | Aᵀ]
- **Geometry**: Torus mapping where indices correspond to physical qubits
- **Parameters**: Arbitrary (L, M) choices yield codes with N = 2*L*M physical qubits
- **Example**: The Gross Code [[144, 12, 12]] uses L=12, M=6

### Simulation Strategy

1. **Erasure Generation**: Random qubit erasures based on erasure probability
2. **Noise Application**: Erased qubits get random Pauli X errors (50% probability)
3. **Hardware Flagging**: Decoder informed of erasures via channel probabilities
4. **Decoding**: BP-OSD decoder attempts error correction
5. **Validation**: Check for logical errors by verifying stabilizer and logical operator conditions

### Performance Optimizations

- **Sparse Matrices**: All operations use sparse matrix representations
- **Parallel Processing**: Work distributed across CPU cores
- **Decoder Reuse**: Expensive decoder initialization done once per worker
- **Vectorization**: All 144 qubits handled in single numpy arrays

## 🧪 Testing

Run the test suite:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest --cov=qldpc tests/
```

## 🐛 Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'qldpc'`

**Solution**:
```bash
# Option 1: Install in editable mode
pip install -e .

# Option 2: Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Memory Issues

**Problem**: `MemoryError` or system slowdown during large simulations

**Solution**:
```python
# Reduce number of cores
simulator = QuantumSimulator(code, num_cores=4)

# Reduce shots
results = simulator.run_experiment(rates, total_shots=10000)
```

### Slow Performance

**Problem**: Simulations taking too long

**Solution**:
```python
# 1. Lower OSD order (faster but less accurate)
config = DecoderConfig(osd_order=5)

# 2. Reduce max BP iterations
config = DecoderConfig(max_iter=30)

# 3. Use adaptive mode for threshold finding
# (automatically stops when threshold is found)
python examples/sweep_scaling.py --adaptive
```

### Decoder Warnings

**Problem**: `UserWarning: old syntax for the 'bposd_decoder'`

**Solution**: This is expected and harmless. The warning is suppressed in the code but may appear in some environments. Update `bposd` to the latest version:
```bash
pip install --upgrade bposd
```

### Invalid Code Sizes (K=0)

**Problem**: Simulation runs but results don't make sense

**Solution**: Some (L, M) choices produce codes with K=0 (no logical qubits). Use the helper:
```bash
python examples/find_valid_sizes.py --L 6 40 --M 3 20
```

### Plotting Issues

**Problem**: `ModuleNotFoundError: No module named 'matplotlib'`

**Solution**:
```bash
pip install matplotlib
```

For headless systems (no display):
```python
# Add to start of plot script
import matplotlib
matplotlib.use('Agg')
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

```bash
# Clone and install in editable mode
git clone https://github.com/tusharpandey13/QLDPC.git
cd QLDPC
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black src/ examples/ tests/
```

## 📚 References

### Bivariate Bicycle Codes
- **Bravyi, S., Cross, A. W., Gambetta, J. M., Maslov, D., Rall, P., & Yoder, T. J.** (2024). "High-threshold and low-overhead fault-tolerant quantum memory." *Nature*, 627, 778-782. [arXiv:2308.07915](https://arxiv.org/abs/2308.07915)

### Decoder
- **Roffe, J., White, D. R., Burton, S., & Campbell, E.** (2020). "Decoding across the quantum low-density parity-check code landscape." *Physical Review Research*, 2(4), 043423. [arXiv:2005.07016](https://arxiv.org/abs/2005.07016)
- **Kovalev, A. A., & Pryadko, L. P.** (2013). "Improved quantum hypergraph-product LDPC codes." *IEEE Transactions on Information Theory*, 59(12), 8318-8330.

### Quantum Error Correction
- **Nielsen, M. A., & Chuang, I. L.** (2010). *Quantum Computation and Quantum Information*. Cambridge University Press.
- **Gottesman, D.** (1997). "Stabilizer codes and quantum error correction." PhD thesis, Caltech. [arXiv:quant-ph/9705052](https://arxiv.org/abs/quant-ph/9705052)

## 🙏 Acknowledgments

- The quantum LDPC community for groundbreaking code constructions
- Developers of the `bposd` and `ldpc` Python packages
- PyMatching for fast MWPM decoding
- IBM Quantum and Google Quantum AI for inspiring experimental demonstrations

## 📧 Contact & Citation

**Author**: Tushar Pandey  
**Email**: tusharp@tamu.edu  
**Institution**: Texas A&M University

### Citation

If you use this code in your research, please cite:

```bibtex
@software{pandey2025qldpc,
  author = {Pandey, Tushar},
  title = {QLDPC: Quantum Low-Density Parity-Check Code Simulation Framework},
  year = {2025},
  url = {https://github.com/tusharpandey13/QLDPC},
  version = {0.1.0}
}
```

### Related Publications

*(Add your papers here when published)*

---

**Note**: This is a research tool for quantum error correction simulation. For production quantum computing applications, additional considerations and validations are required.

## 🌟 Star History

If you find this project useful, please consider giving it a star ⭐ on GitHub!

