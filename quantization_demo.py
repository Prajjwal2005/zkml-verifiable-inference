"""
Step 4: Quantization Demonstration - Show how floating-point weights
get converted to finite field elements.

This script walks through the exact process EZKL performs internally,
making the abstract math from the report concrete and runnable.
"""

import torch
import numpy as np
import os
import json


# The BN254 scalar field prime (used by Halo2/EZKL)
BN254_PRIME = 21888242871839275222246405745257275088548364400416034343698204186575808495617

# Scale factor: 2^16 = 65536 (16 fractional bits)
SCALE = 2 ** 16


def demonstrate_quantization():
    print("=" * 60)
    print("QUANTIZATION WALKTHROUGH")
    print("How floating-point weights become finite field elements")
    print("=" * 60)

    # Load the actual trained model weights
    from train_model import SmallCNN
    model = SmallCNN()
    model.load_state_dict(torch.load(
        os.path.join("models", "mnist_cnn.pth"),
        map_location="cpu",
        weights_only=True
    ))

    # Grab a few actual weights from the first conv layer
    weights = model.conv1.weight.data.flatten()[:5].numpy()

    print(f"\nScale factor S = 2^16 = {SCALE}")
    print(f"Field prime p = {BN254_PRIME}")
    print(f"            p has {len(str(BN254_PRIME))} digits ({BN254_PRIME.bit_length()} bits)")

    print(f"\n{'='*60}")
    print("STEP-BY-STEP: Converting first 5 weights from conv1")
    print(f"{'='*60}")

    quantized_values = []
    for i, w in enumerate(weights):
        print(f"\n--- Weight {i}: w = {w:.6f} ---")

        # Step 1: Scale
        w_scaled = w * SCALE
        print(f"  Step 1 (scale):    w * {SCALE} = {w_scaled:.2f}")

        # Step 2: Round to nearest integer
        w_int = int(round(w_scaled))
        print(f"  Step 2 (round):    round({w_scaled:.2f}) = {w_int}")

        # Step 3: Map to field element
        # Negative numbers wrap around: -k maps to p - k
        if w_int >= 0:
            w_field = w_int % BN254_PRIME
            print(f"  Step 3 (to field): {w_int} is positive, so field element = {w_int}")
        else:
            w_field = (BN254_PRIME + w_int) % BN254_PRIME
            print(f"  Step 3 (to field): {w_int} is negative, so field element = p + ({w_int})")
            print(f"                     = {w_field}")

        # Step 4: Show the reconstruction error
        w_reconstructed = w_int / SCALE
        error = abs(w - w_reconstructed)
        print(f"  Step 4 (verify):   {w_int} / {SCALE} = {w_reconstructed:.6f}")
        print(f"  Quantization error: |{w:.6f} - {w_reconstructed:.6f}| = {error:.8f}")

        quantized_values.append(w_field)

    # Now demonstrate arithmetic inside the field
    print(f"\n{'='*60}")
    print("ARITHMETIC INSIDE THE FIELD")
    print(f"{'='*60}")

    if len(quantized_values) >= 2:
        a = int(round(weights[0] * SCALE))
        b = int(round(weights[1] * SCALE))

        print(f"\n  w0 (quantized) = {a}")
        print(f"  w1 (quantized) = {b}")

        # Addition in the field
        field_sum = (a + b) % BN254_PRIME
        real_sum = weights[0] + weights[1]
        reconstructed_sum = (a + b) / SCALE
        print(f"\n  Addition:")
        print(f"    Float:   {weights[0]:.6f} + {weights[1]:.6f} = {real_sum:.6f}")
        print(f"    Field:   {a} + {b} = {a + b}")
        print(f"    Reconstruct: {a + b} / {SCALE} = {reconstructed_sum:.6f}")
        print(f"    Error:   {abs(real_sum - reconstructed_sum):.8f}")

        # Multiplication in the field (this is where rescaling happens)
        product = a * b
        # After multiplication, we have double the fractional bits (32 instead of 16)
        # So we must divide by SCALE to get back to 16 fractional bits
        rescaled = product // SCALE  # integer division (truncation)
        real_product = weights[0] * weights[1]
        reconstructed_product = rescaled / SCALE
        print(f"\n  Multiplication:")
        print(f"    Float:   {weights[0]:.6f} * {weights[1]:.6f} = {real_product:.6f}")
        print(f"    Field:   {a} * {b} = {product}")
        print(f"    Rescale: {product} // {SCALE} = {rescaled}")
        print(f"    Reconstruct: {rescaled} / {SCALE} = {reconstructed_product:.6f}")
        print(f"    Error:   {abs(real_product - reconstructed_product):.8f}")

    # Show accuracy impact statistics
    print(f"\n{'='*60}")
    print("QUANTIZATION ERROR STATISTICS (all model weights)")
    print(f"{'='*60}")

    all_errors = []
    for name, param in model.named_parameters():
        flat = param.data.flatten().numpy()
        quantized = np.round(flat * SCALE).astype(np.int64)
        reconstructed = quantized / SCALE
        errors = np.abs(flat - reconstructed)
        all_errors.extend(errors.tolist())

    all_errors = np.array(all_errors)
    print(f"\n  Total weights analyzed: {len(all_errors):,}")
    print(f"  Max quantization error:  {all_errors.max():.8f}")
    print(f"  Mean quantization error: {all_errors.mean():.8f}")
    print(f"  Std quantization error:  {all_errors.std():.8f}")
    print(f"  Theoretical max error:   {1/(2*SCALE):.8f} (= 1/(2*S))")
    print(f"\n  With 16-bit fixed-point (S=65536), the maximum possible")
    print(f"  quantization error per weight is {1/(2*SCALE):.8f}.")
    print(f"  For our {len(all_errors):,} parameters, the accumulated error")
    print(f"  is small enough that MNIST accuracy typically drops by < 1%.")


if __name__ == "__main__":
    demonstrate_quantization()
