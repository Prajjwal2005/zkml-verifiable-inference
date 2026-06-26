# Architecture & Design: ZKML Verifiable Inference

This document provides a detailed overview of the architectural components, design decisions, and cryptographic processes that power the ZKML (Zero-Knowledge Machine Learning) verifiable inference pipeline.

## 1. Overview of the Pipeline

The project implements an end-to-end Zero-Knowledge SNARK (zk-SNARK) pipeline for a convolutional neural network. It proves that a specific machine learning model executed inference on a specific input correctly, yielding a specific output, without revealing the model's internal weights or intermediate activations to the verifier.

The pipeline comprises four distinct phases:
1. **Training & Export:** Training a minimalist CNN on the MNIST dataset and exporting it to an ONNX graph.
2. **Preprocessing & Calibration:** Setting up the EZKL proving environment, calibrating scaling factors for quantization, and bounding the circuit size.
3. **Cryptographic Setup:** Compiling the computational graph into a cryptographic circuit and generating the required keys using a KZG Structured Reference String (SRS).
4. **Proving & Verifying:** Generating the witness (execution trace), producing the zk-SNARK proof, and verifying it in milliseconds.

---

## 2. Neural Network Architecture

The model is defined in `train_model.py`. To make zero-knowledge proving computationally feasible on consumer hardware, the model must be extremely small. Typical deep neural networks require massive amounts of RAM and time to prove due to the large number of arithmetic constraints they produce.

### Model Specs (`SmallCNN`)
- **Input:** 1x28x28 grayscale image (MNIST).
- **Layer 1:** 2D Convolution (4 filters, 3x3 kernel, stride 2, padding 1) $\rightarrow$ ReLU Activation. Output size: 4x14x14.
- **Layer 2:** 2D Convolution (8 filters, 3x3 kernel, stride 2, padding 1) $\rightarrow$ ReLU Activation. Output size: 8x7x7.
- **Layer 3:** Flatten layer followed by a Fully Connected (Linear) layer. Input: 392 (8*7*7), Output: 10 (digits 0-9).
- **Total Parameters:** ~4,300.

### Key Design Constraints
- **Strided Convolutions vs. Max Pooling:** While Max Pooling is supported in EZKL, strided convolutions are used here to reduce the non-linear constraints and keep the arithmetic circuit simple and predictable.
- **No Batch Normalization:** BatchNorm relies on running statistics that change between training and inference modes. It severely complicates quantization for ZK circuits and was thus avoided.
- **Activation Function:** ReLU is used exclusively because it can be efficiently implemented in zero-knowledge circuits using lookup tables.

---

## 3. Cryptographic Framework & EZKL Engine

The heavy lifting of the ZK proofs is handled by **EZKL**, a zero-knowledge machine learning engine. It translates standard ONNX computational graphs into Halo2/PLONKish arithmetization circuits.

### Quantization & Field Arithmetic (`quantization_demo.py`)
Neural networks typically compute using 32-bit floating-point numbers (`float32`). Cryptographic circuits, however, only operate over finite fields (integers modulo a large prime). 
- **Fixed-Point Arithmetic:** The weights and activations are quantized into fixed-point representations. 
- **Scaling Factors:** A scaling factor is calibrated during the pipeline's setup phase to preserve precision without overflowing the bounds of the finite field.

### Circuit Compilation (`compile_circuit`)
The EZKL engine walks the ONNX graph and converts mathematical operations (multiplications, additions) into structural polynomial constraints (gates). Non-linear operations like ReLU are typically handled using cryptographic lookup tables (Halo2's lookup argument).

---

## 4. The ZK-SNARK Pipeline (`zkml_pipeline.py`)

The end-to-end execution follows these sequential steps:

1. **`gen_settings` & `calibrate_settings`**:
   - Parses the ONNX model and generates an initial configuration (`settings.json`).
   - Calibrates the settings over sample inputs. It determines the optimal `logrows`—a measure of the circuit size, essentially mapping to $2^{\text{logrows}}$ rows in the PLONKish table.

2. **`compile_circuit`**:
   - Converts the ONNX model into a compiled `.ezkl` circuit file, mapping the operations to Halo2 gates and lookup tables.

3. **`get_srs`**:
   - Fetches the Structured Reference String (SRS) for the KZG polynomial commitment scheme. The size of the SRS depends directly on the required `logrows`.

4. **`setup`**:
   - Executes the trusted setup (or processes the universal setup) to generate two distinct keys:
     - **Proving Key (`pk.key`)**: A massive file (~264 MB) containing polynomials required by the prover to construct the proof.
     - **Verifying Key (`vk.key`)**: A tiny file (~130 KB) containing cryptographic commitments used by the verifier to validate the proof in milliseconds.

5. **`gen_witness`**:
   - Performs "in-circuit inference". It evaluates the computational graph using the quantized inputs and weights, generating a full execution trace (the "witness") of every intermediate variable in the neural network.

6. **`prove`**:
   - The prover uses the witness, the proving key, and the compiled circuit to generate a succinct cryptographic proof (`proof.json`). This step is the bottleneck, requiring significant CPU compute and memory.

7. **`verify`**:
   - The verifier takes the tiny proof, the verifying key, the public inputs, and the public outputs. Through pairing-based cryptography, the verifier confirms that the outputs were genuinely produced by executing the specific circuit on the inputs.

---

## 5. Model Commitment (`model_commitment.py`)

To ensure **verifiable inference**, the verifier must be certain that the prover didn't just run *some* valid network, but the *exact* neural network agreed upon.
- A cryptographic hash (e.g., Poseidon or Keccak) of the model's weights and architecture is computed.
- This hash serves as a unique fingerprint. When the ZK proof is verified, the verifier checks this commitment to guarantee no weight tampering occurred during the proving process.
