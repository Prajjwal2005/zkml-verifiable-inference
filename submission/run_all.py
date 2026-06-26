"""
Main entry point: Run the entire ZKML pipeline end-to-end.

Usage:
    cd C:/ZKML_Project
    python run_all.py

This script runs each step in order:
1. Train the CNN and export to ONNX
2. Compute model commitment H(theta)
3. Demonstrate quantization process
4. Run the full EZKL pipeline (settings, compile, SRS, setup, witness, prove, verify)
5. Print a final summary of all benchmarks
"""

import os
import sys
import json
import time
import asyncio


def run_step(description, module_name, function_name=None):
    """Run a step and handle errors gracefully."""
    print(f"\n{'#' * 70}")
    print(f"# {description}")
    print(f"{'#' * 70}\n")

    try:
        module = __import__(module_name)
        if function_name:
            func = getattr(module, function_name)
            func()
        return True
    except Exception as e:
        print(f"\nERROR in {module_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_async_step(description, module_name, function_name):
    """Run an async step."""
    print(f"\n{'#' * 70}")
    print(f"# {description}")
    print(f"{'#' * 70}\n")

    try:
        module = __import__(module_name)
        func = getattr(module, function_name)
        result = await func()
        return result
    except Exception as e:
        print(f"\nERROR in {module_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


async def main():
    overall_start = time.time()

    print("=" * 70)
    print("   ZKML PROJECT: Verifiable ML Inference using Zero-Knowledge Proofs")
    print("   BITS F463 - Cryptography | Phase III Prototype")
    print("=" * 70)

    # Step 1: Train model
    success = run_step(
        "STEP 1/4: Training CNN on MNIST and exporting to ONNX",
        "train_model",
        "train_model"
    )
    if not success:
        print("\nTraining failed. Stopping.")
        return

    # Step 2: Model commitment
    run_step(
        "STEP 2/4: Computing model commitment H(theta)",
        "model_commitment",
        "demonstrate_commitment"
    )

    # Step 3: Quantization demo
    run_step(
        "STEP 3/4: Demonstrating quantization process",
        "quantization_demo",
        "demonstrate_quantization"
    )

    # Step 4: EZKL pipeline (7 internal steps: settings, compile, SRS, setup, witness, prove, verify)
    benchmarks = await run_async_step(
        "STEP 4/4: Running EZKL ZK proof pipeline",
        "zkml_pipeline",
        "run_pipeline"
    )

    # Final summary
    overall_time = time.time() - overall_start

    print(f"\n\n{'=' * 70}")
    print("   FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n  Total pipeline time: {overall_time:.1f} seconds")

    # Load metadata
    metadata_path = os.path.join("models", "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
        print(f"\n  Model:")
        print(f"    Parameters:       {metadata['total_params']:,}")
        print(f"    Test accuracy:    {metadata['test_accuracy']:.2f}%")
        print(f"    Training time:    {metadata['training_time_seconds']}s")

    if benchmarks:
        print(f"\n  ZK Proof Pipeline:")
        print(f"    Settings:         {benchmarks.get('settings_sec', 'N/A')}s")
        print(f"    Circuit compile:  {benchmarks.get('compilation_sec', 'N/A')}s")
        print(f"    SRS download:     {benchmarks.get('srs_sec', 'N/A')}s")
        print(f"    Setup (keygen):   {benchmarks.get('setup_sec', 'N/A')}s")
        print(f"    Witness gen:      {benchmarks.get('witness_sec', 'N/A')}s")
        print(f"    Proof generation: {benchmarks.get('proving_sec', 'N/A')}s")
        print(f"    Verification:     {benchmarks.get('verification_ms', 'N/A')}ms")
        print(f"    Proof size:       {benchmarks.get('proof_size_kb', 'N/A')} KB")
        print(f"    Verified:         {benchmarks.get('verified', 'N/A')}")

    print(f"\n  Output files:")
    print(f"    models/mnist_cnn.onnx      - ONNX model")
    print(f"    models/commitment.json     - H(theta) commitment")
    print(f"    proofs/settings.json       - Circuit settings")
    print(f"    proofs/compiled.ezkl       - Compiled circuit")
    print(f"    proofs/pk.key              - Proving key")
    print(f"    proofs/vk.key              - Verification key")
    print(f"    proofs/witness.json        - Witness")
    print(f"    proofs/proof.json          - ZK proof (pi)")
    print(f"    proofs/benchmarks.json     - All timing data")
    print(f"\n{'=' * 70}")
    print("   Pipeline complete. All results saved.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    asyncio.run(main())