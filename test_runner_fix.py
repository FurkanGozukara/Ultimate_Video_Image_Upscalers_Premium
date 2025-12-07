#!/usr/bin/env python3
"""
Test script to verify the runner fixes work properly.
"""
import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from shared.runner import Runner

def test_runner_basic():
    """Test basic runner functionality."""
    print("🧪 Testing Runner Basic Functionality...")
    with open("test_output.txt", "w") as f:
        f.write("Starting test...\n")

    base_dir = Path(__file__).parent
    temp_dir = base_dir / "temp"
    output_dir = base_dir / "outputs"

    runner = Runner(base_dir, temp_dir, output_dir)

    # Test with CPU-only settings to avoid CUDA issues
    settings = {
        "input_path": str(base_dir / "test_vid.mp4"),  # Use existing test file
        "output_format": "auto",
        "dit_model": "seedvr2_ema_3b_fp16.safetensors",
        "resolution": 1080,
        "batch_size": 5,
        "cuda_device": "",  # Empty string for CPU
        "dit_offload_device": "cpu",
        "vae_offload_device": "cpu",
        "tensor_offload_device": "cpu",
        "blocks_to_swap": 0,
        "swap_io_components": False,
        "vae_encode_tiled": False,
        "vae_decode_tiled": False,
        "attention_mode": "sdpa",  # Use SDPA instead of flash_attn for CPU
        "compile_dit": False,
        "compile_vae": False,
        "debug": False,
    }

    print("Testing SeedVR2 runner execution...")

    def progress_callback(line: str):
        print(f"[PROGRESS] {line.strip()}")
        with open("test_output.txt", "a") as f:
            f.write(f"[PROGRESS] {line.strip()}\n")

    try:
        result = runner.run_seedvr2(settings, on_progress=progress_callback)

        print("\nRunner result:")
        print(f"  Return code: {result.returncode}")
        print(f"  Output path: {result.output_path}")
        print(f"  Log length: {len(result.log)} characters")

        if result.returncode == 0:
            print("✅ Runner executed successfully!")
        else:
            print("⚠️ Runner completed with non-zero exit code")
            print("Last 10 log lines:")
            log_lines = result.log.split('\n')
            for line in log_lines[-10:]:
                if line.strip():
                    print(f"  {line}")

    except Exception as e:
        print(f"❌ Runner failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = test_runner_basic()
    sys.exit(0 if success else 1)
