#!/usr/bin/env python3
"""
Test script for the preset system functionality.
"""
import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from shared.preset_manager import PresetManager

def test_preset_system():
    """Test the preset system functionality."""
    print("🧪 Testing Preset System...")

    # Create a test presets directory
    test_dir = Path("test_presets")
    test_dir.mkdir(exist_ok=True)

    try:
        # Initialize preset manager
        pm = PresetManager(test_dir)
        print("✅ PresetManager initialized")

        # Test saving a preset
        test_preset = {
            "batch_size": 5,
            "resolution": 1080,
            "color_correction": "lab",
            "cuda_device": "0"
        }

        result = pm.save_preset_safe("seedvr2", "test_model", "test_preset", test_preset)
        print(f"✅ Preset saved: {result}")

        # Test loading the preset
        loaded = pm.load_preset_safe("seedvr2", "test_model", "test_preset")
        print(f"✅ Preset loaded: {loaded}")

        # Test listing presets
        presets = pm.list_presets("seedvr2", "test_model")
        print(f"✅ Presets listed: {presets}")

        # Test last used tracking
        pm.set_last_used("seedvr2", "test_model", "test_preset")
        last_used = pm.get_last_used_name("seedvr2", "test_model")
        print(f"✅ Last used set: {last_used}")

        # Test loading last used
        last_preset = pm.load_last_used("seedvr2", "test_model")
        print(f"✅ Last used loaded: {last_preset is not None}")

        # Test validation
        validated = pm.validate_preset_constraints(test_preset, "seedvr2", "test_model")
        print(f"✅ Validation applied: {validated == test_preset}")

        print("🎉 All preset system tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    finally:
        # Clean up
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)

    return True

if __name__ == "__main__":
    success = test_preset_system()
    sys.exit(0 if success else 1)
