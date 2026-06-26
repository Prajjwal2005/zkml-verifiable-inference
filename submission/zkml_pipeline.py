"""
Step 4: Full EZKL ZK proof pipeline.

Pipeline steps:
1. gen_settings    - create circuit config from ONNX model
2. calibrate       - auto-tune quantization and logrows
3. compile_circuit - ONNX -> arithmetic circuit
4. get_srs         - download structured reference string
5. setup           - generate proving key (pk) and verification key (vk)
6. gen_witness     - run quantized inference, record execution trace
7. prove           - generate ZK proof
8. verify          - check proof validity
"""

import os
import json
import time
import asyncio
import inspect
import ezkl


async def maybe_await(result):
    """Handle EZKL functions that may be sync or async depending on version."""
    if inspect.isawaitable(result):
        return await result
    return result


# Paths
MODEL_PATH = os.path.join("models", "mnist_cnn.onnx")
INPUT_PATH = os.path.join("models", "test_input.json")
OUTPUT_DIR = "proofs"

SETTINGS_PATH = os.path.join(OUTPUT_DIR, "settings.json")
COMPILED_PATH = os.path.join(OUTPUT_DIR, "compiled.ezkl")
SRS_PATH = os.path.join(OUTPUT_DIR, "kzg.srs")
PK_PATH = os.path.join(OUTPUT_DIR, "pk.key")
VK_PATH = os.path.join(OUTPUT_DIR, "vk.key")
WITNESS_PATH = os.path.join(OUTPUT_DIR, "witness.json")
PROOF_PATH = os.path.join(OUTPUT_DIR, "proof.json")


async def run_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    benchmarks = {}

    # ==================== STEP 1: GEN SETTINGS ====================
    print("=" * 60)
    print("STEP 1: Generating circuit settings...")
    print("=" * 60)
    t = time.time()

    res = await maybe_await(ezkl.gen_settings(MODEL_PATH, SETTINGS_PATH))
    assert res, "gen_settings failed"

    benchmarks["settings_sec"] = round(time.time() - t, 2)
    print(f"  Done in {benchmarks['settings_sec']}s")

    # ==================== STEP 2: CALIBRATE ====================
    print("\n" + "=" * 60)
    print("STEP 2: Calibrating settings (auto-tuning quantization)...")
    print("=" * 60)
    t = time.time()

    res = await maybe_await(ezkl.calibrate_settings(
        INPUT_PATH, MODEL_PATH, SETTINGS_PATH, "resources"
    ))
    assert res, "calibrate_settings failed"

    benchmarks["calibrate_sec"] = round(time.time() - t, 2)
    print(f"  Done in {benchmarks['calibrate_sec']}s")

    # Print calibrated logrows
    with open(SETTINGS_PATH) as f:
        settings = json.load(f)
    logrows = settings.get("run_args", {}).get("logrows", "?")
    print(f"  Calibrated logrows: {logrows}")

    # ==================== STEP 3: COMPILE CIRCUIT ====================
    print("\n" + "=" * 60)
    print("STEP 3: Compiling ONNX to arithmetic circuit...")
    print("=" * 60)
    t = time.time()

    res = await maybe_await(ezkl.compile_circuit(MODEL_PATH, COMPILED_PATH, SETTINGS_PATH))
    assert res, "compile_circuit failed"

    benchmarks["compile_sec"] = round(time.time() - t, 2)
    compiled_kb = os.path.getsize(COMPILED_PATH) / 1024
    print(f"  Done in {benchmarks['compile_sec']}s")
    print(f"  Compiled circuit: {compiled_kb:.1f} KB")

    # ==================== STEP 4: GET SRS ====================
    print("\n" + "=" * 60)
    print("STEP 4: Downloading SRS (structured reference string)...")
    print("=" * 60)
    t = time.time()

    # Fix for Windows: EZKL's Rust code uses HOME env var (Unix convention)
    if "HOME" not in os.environ and "USERPROFILE" in os.environ:
        os.environ["HOME"] = os.environ["USERPROFILE"]

    with open(SETTINGS_PATH) as f:
        settings = json.load(f)
    logrows = settings["run_args"]["logrows"]
    print(f"  Fetching SRS for logrows={logrows}...")
    res = await maybe_await(ezkl.get_srs(SRS_PATH, logrows))

    # EZKL may cache SRS in its own directory instead of our path.
    # Search for it if our path is empty.
    if not os.path.exists(SRS_PATH) or os.path.getsize(SRS_PATH) == 0:
        home = os.environ.get("HOME", os.environ.get("USERPROFILE", ""))
        search_dirs = [
            os.path.join(home, ".ezkl"),
            os.path.join(home, ".ezkl", "srs"),
            home,
        ]
        import glob
        for d in search_dirs:
            if not os.path.isdir(d):
                continue
            for pattern in [f"*{logrows}*", "*.srs", "kzg*"]:
                matches = glob.glob(os.path.join(d, pattern))
                for m in matches:
                    if os.path.getsize(m) > 1000:
                        print(f"  Found cached SRS at: {m}")
                        import shutil
                        shutil.copy2(m, SRS_PATH)
                        break
                if os.path.exists(SRS_PATH) and os.path.getsize(SRS_PATH) > 1000:
                    break
            if os.path.exists(SRS_PATH) and os.path.getsize(SRS_PATH) > 1000:
                break

    # If still not found, set SRS_PATH to None so EZKL uses its internal cache
    if not os.path.exists(SRS_PATH) or os.path.getsize(SRS_PATH) == 0:
        print("  SRS not at expected path. Will let EZKL use its internal cache.")
        SRS_ACTUAL = None
    else:
        SRS_ACTUAL = SRS_PATH

    benchmarks["srs_sec"] = round(time.time() - t, 2)
    if os.path.exists(SRS_PATH):
        srs_mb = os.path.getsize(SRS_PATH) / (1024 * 1024)
        print(f"  Done in {benchmarks['srs_sec']}s")
        print(f"  SRS size: {srs_mb:.1f} MB")
    else:
        print(f"  WARNING: SRS file not found at {SRS_PATH}")

    # ==================== STEP 5: SETUP (KEY GEN) ====================
    print("\n" + "=" * 60)
    print("STEP 5: Running setup (generating pk and vk)...")
    print("=" * 60)
    t = time.time()

    res = await maybe_await(ezkl.setup(COMPILED_PATH, VK_PATH, PK_PATH, srs_path=SRS_ACTUAL))
    assert res, "setup failed"

    benchmarks["setup_sec"] = round(time.time() - t, 2)
    pk_mb = os.path.getsize(PK_PATH) / (1024 * 1024)
    vk_kb = os.path.getsize(VK_PATH) / 1024
    print(f"  Done in {benchmarks['setup_sec']}s")
    print(f"  Proving key:      {pk_mb:.2f} MB")
    print(f"  Verification key: {vk_kb:.2f} KB")

    # ==================== STEP 6: GEN WITNESS ====================
    print("\n" + "=" * 60)
    print("STEP 6: Generating witness (in-circuit inference)...")
    print("=" * 60)
    t = time.time()

    res = await maybe_await(ezkl.gen_witness(INPUT_PATH, COMPILED_PATH, WITNESS_PATH))
    assert os.path.exists(WITNESS_PATH), "gen_witness failed"

    benchmarks["witness_sec"] = round(time.time() - t, 2)
    print(f"  Done in {benchmarks['witness_sec']}s")

    # ==================== STEP 7: PROVE ====================
    print("\n" + "=" * 60)
    print("STEP 7: Generating ZK proof (this is the slow step)...")
    print("=" * 60)
    t = time.time()

    res = await maybe_await(ezkl.prove(
        WITNESS_PATH, COMPILED_PATH, PK_PATH, PROOF_PATH, srs_path=SRS_ACTUAL
    ))
    assert os.path.exists(PROOF_PATH), "prove failed"

    benchmarks["proving_sec"] = round(time.time() - t, 2)
    proof_kb = os.path.getsize(PROOF_PATH) / 1024
    benchmarks["proof_size_kb"] = round(proof_kb, 2)
    print(f"  Done in {benchmarks['proving_sec']}s")
    print(f"  Proof size: {proof_kb:.2f} KB")

    # ==================== STEP 8: VERIFY ====================
    print("\n" + "=" * 60)
    print("STEP 8: Verifying proof...")
    print("=" * 60)
    t = time.time()

    res = await maybe_await(ezkl.verify(PROOF_PATH, SETTINGS_PATH, VK_PATH, srs_path=SRS_ACTUAL))

    benchmarks["verification_ms"] = round((time.time() - t) * 1000, 2)
    benchmarks["verified"] = bool(res)

    if res:
        print(f"  PROOF VERIFIED SUCCESSFULLY in {benchmarks['verification_ms']}ms")
    else:
        print(f"  PROOF VERIFICATION FAILED")

    # Save benchmarks
    benchmarks["pk_size_mb"] = round(pk_mb, 2)
    benchmarks["vk_size_kb"] = round(vk_kb, 2)
    with open(os.path.join(OUTPUT_DIR, "benchmarks.json"), "w") as f:
        json.dump(benchmarks, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    for key, val in benchmarks.items():
        print(f"  {key}: {val}")

    return benchmarks


if __name__ == "__main__":
    result = asyncio.run(run_pipeline())
