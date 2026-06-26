"""
Step 3: Model Commitment - Compute H(theta) and demonstrate the binding property.

This script shows the cryptographic commitment concept:
1. Load the trained model weights
2. Hash them using SHA-256 (for the external/out-of-circuit commitment)
3. Show that changing even one weight completely changes the hash
4. This is what makes model substitution attacks detectable

In the actual ZKP circuit, Poseidon hash is used instead (it is
ZKP-friendly), but the concept is identical.
"""

import torch
import hashlib
import json
import os
import numpy as np
import copy


def compute_model_hash(state_dict):
    """
    Compute SHA-256 hash of all model parameters.
    We serialize all weights into a byte string and hash it.
    """
    hasher = hashlib.sha256()
    for name in sorted(state_dict.keys()):
        # Convert each parameter tensor to bytes
        param_bytes = state_dict[name].cpu().numpy().tobytes()
        hasher.update(param_bytes)
    return hasher.hexdigest()


def demonstrate_commitment():
    print("=" * 60)
    print("MODEL COMMITMENT DEMONSTRATION")
    print("=" * 60)

    # Load the trained model
    from train_model import SmallCNN
    model = SmallCNN()
    model.load_state_dict(torch.load(
        os.path.join("models", "mnist_cnn.pth"),
        map_location="cpu",
        weights_only=True
    ))
    model.eval()

    # Compute commitment hash H(theta)
    original_hash = compute_model_hash(model.state_dict())
    print(f"\nOriginal model hash H(theta):")
    print(f"  {original_hash}")

    # Show total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Now demonstrate the binding property:
    # Change ONE weight by a tiny amount and show the hash completely changes
    print("\n" + "-" * 60)
    print("BINDING PROPERTY DEMONSTRATION")
    print("-" * 60)

    tampered_state = copy.deepcopy(model.state_dict())

    # Find the first weight and modify it slightly
    first_key = sorted(tampered_state.keys())[0]
    original_value = tampered_state[first_key].flatten()[0].item()

    # Add a tiny perturbation (0.0001)
    tampered_state[first_key].flatten()[0] += 0.0001
    new_value = tampered_state[first_key].flatten()[0].item()

    tampered_hash = compute_model_hash(tampered_state)

    print(f"\n  Modified parameter: {first_key}[0]")
    print(f"  Original value:     {original_value:.6f}")
    print(f"  Tampered value:     {new_value:.6f}")
    print(f"  Difference:         {abs(new_value - original_value):.6f}")
    print(f"\n  Original hash:  {original_hash}")
    print(f"  Tampered hash:  {tampered_hash}")
    print(f"\n  Hashes match? {original_hash == tampered_hash}")

    # Count how many hex characters differ
    diff_count = sum(1 for a, b in zip(original_hash, tampered_hash) if a != b)
    print(f"  Characters that differ: {diff_count} out of {len(original_hash)}")
    print(f"\n  This demonstrates the avalanche effect: changing one weight by")
    print(f"  0.0001 causes {diff_count}/64 hex characters to change.")
    print(f"  A dishonest provider cannot substitute a different model without")
    print(f"  detection because the hash commitment would not match.")

    # Save the commitment
    commitment = {
        "model_hash_sha256": original_hash,
        "total_parameters": total_params,
        "hash_algorithm": "SHA-256",
        "note": "In the ZKP circuit, Poseidon hash is used instead for efficiency"
    }
    commitment_path = os.path.join("models", "commitment.json")
    with open(commitment_path, "w") as f:
        json.dump(commitment, f, indent=2)
    print(f"\nCommitment saved to: {commitment_path}")


if __name__ == "__main__":
    demonstrate_commitment()
