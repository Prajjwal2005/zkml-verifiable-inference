# ZKML Project: Verifiable ML Inference using Zero-Knowledge Proofs

## BITS F463 - Cryptography | Phase III Prototype
**Author:** Prajjwal (2023A7PS0381U)

## What This Project Does

This prototype demonstrates a complete pipeline where:
1. A neural network is trained on MNIST handwritten digits
2. The model is exported and compiled into a zero-knowledge arithmetic circuit
3. A cryptographic proof is generated proving the inference was done correctly
4. Anyone can verify that proof in milliseconds without seeing the model weights

## Project Structure

```
ZKML_Project/
    train_model.py         - Train CNN, export to ONNX
    model_commitment.py    - Compute H(theta) and demonstrate binding
    quantization_demo.py   - Walk through float-to-field conversion
    zkml_pipeline.py       - Full EZKL pipeline (compile, setup, prove, verify)
    run_all.py             - Orchestrator: runs everything end-to-end
    requirements.txt       - Python dependencies
    models/                - Trained model, ONNX file, test input
    proofs/                - Circuit, keys, proof, benchmarks
    data/                  - MNIST dataset (auto-downloaded)
```

## Setup Instructions

### 1. Create a Python virtual environment
```bash
cd C:\ZKML_Project
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install onnx numpy ezkl
```

Note: The torch install command above is for CUDA 12.4 (RTX 5070).
If you want CPU only, just run: `pip install torch torchvision`

### 3. Run the full pipeline
```bash
python run_all.py
```

Or run individual steps:
```bash
python train_model.py          # Step 1: Train and export
python model_commitment.py     # Step 2: Hash commitment demo
python quantization_demo.py    # Step 3: Quantization walkthrough
python zkml_pipeline.py        # Step 4: Full ZK pipeline
```

## Expected Output

- Training: ~2 minutes (GPU) / ~5 minutes (CPU)
- Test accuracy: 97-98%
- Circuit compilation: seconds to minutes
- Proof generation: minutes (the expensive step)
- Verification: milliseconds
- Proof size: tens of KB

## Troubleshooting

**ezkl installation fails:**
Try `pip install ezkl --no-cache-dir`. On Windows, you may need to
install Visual C++ Build Tools first.

**Out of memory during proving:**
The model is designed to be small (~4300 params), so this should not
happen. If it does, reduce the model further in train_model.py.

**CUDA not detected:**
The training script falls back to CPU automatically. EZKL proving
runs on CPU regardless (GPU acceleration is not yet in the Python bindings).
