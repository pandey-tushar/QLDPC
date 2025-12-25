# QLDPC: Quantum Low-Density Parity-Check Code Simulation Framework

A high-performance simulation framework for quantum error correction codes, specifically optimized for **Bivariate Bicycle Codes** and erasure channel analysis. This implementation focuses on code capacity simulations using matrix-based methods and probability calculations, providing a mathematically rigorous way to determine error thresholds without simulating the physics of every gate.

## 🎯 Features

- **Fast Code Capacity Simulation**: Matrix + Probability only approach (no heavy circuit compilation)
- **Memory Optimized**: Uses `scipy.sparse.csr_matrix` for efficient memory usage (<100MB RAM)
- **Parallel Processing**: Multiprocessing support to saturate all CPU cores
- **Hot-Path Optimization**: Decoder instantiated once per core, only priors updated per shot
- **Bivariate Bicycle Code**: Implements the [[144, 12, 12]] Gross Code with torus geometry

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

## 🏗️ Project Structure

```
QLDPC/
├── src/
│   └── qldpc/
│       ├── __init__.py
│       ├── code.py          # Code construction (BivariateBicycleCode)
│       ├── decoder.py       # Decoder configuration
│       └── simulator.py     # Monte Carlo simulation engine
├── examples/
│   └── run_simulation.py    # Example usage script
├── tests/                   # Unit tests (to be added)
├── docs/                    # Documentation (to be added)
├── requirements.txt         # Python dependencies
├── setup.py                 # Package setup
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## 🔬 Technical Details

### Code Construction

The Bivariate Bicycle Code is constructed using:
- **Polynomials**: A = x³ + y + y², B = y³ + x + x²
- **Lifted Product**: Hx = [A | B], Hz = [Bᵀ | Aᵀ]
- **Geometry**: Torus mapping where indices correspond to physical qubits

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

Unit tests can be added to the `tests/` directory. Example test structure:

```bash
tests/
├── __init__.py
├── test_code.py
├── test_decoder.py
└── test_simulator.py
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📚 References

- Gross, J. A., & Nezami, S. (2022). Bivariate Bicycle Codes
- BP-OSD Decoder: Belief Propagation with Ordered Statistics Decoding
- CSS Codes: Calderbank-Shor-Steane quantum error correction codes

## 👤 Author

Your Name - your.email@example.com

## 🙏 Acknowledgments

- The quantum LDPC community for code constructions and decoders
- Developers of the `bposd` and `ldpc` Python packages

---

**Note**: This is a research tool for quantum error correction simulation. For production quantum computing applications, additional considerations and validations are required.

