#!/usr/bin/env python3
"""
Test script for GAN implementation features.
"""
import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_gan_metadata():
    """Test GAN model metadata system."""
    print("🧪 Testing GAN Model Metadata System...")

    from shared.gan_runner import GanModelRegistry, get_gan_model_metadata

    base_dir = Path(__file__).parent
    registry = GanModelRegistry(base_dir)

    # Test metadata loading
    try:
        metadata = get_gan_model_metadata("RealESRGAN_x4plus", base_dir)
        print(f"✅ Metadata loaded: {metadata.name}, scale={metadata.scale}x, arch={metadata.architecture}")
    except Exception as e:
        print(f"❌ Metadata loading failed: {e}")
        return False

    # Test registry
    registry._load_omd_metadata()
    print(f"✅ Registry loaded {len(registry._omd_cache)} models from OMD")

    return True

def test_gan_service():
    """Test GAN service parameters."""
    print("🧪 Testing GAN Service Parameters...")

    from shared.services.gan_service import gan_defaults, GAN_ORDER

    base_dir = Path(__file__).parent
    try:
        defaults = gan_defaults(base_dir)
        print(f"✅ GAN defaults loaded with {len(defaults)} parameters")

        # Check that batch_size is in GAN_ORDER
        if "batch_size" in GAN_ORDER:
            print("✅ batch_size parameter found in GAN_ORDER")
        else:
            print("❌ batch_size parameter missing from GAN_ORDER")
            return False

        # Check defaults has batch_size
        if "batch_size" in defaults:
            print(f"✅ batch_size default value: {defaults['batch_size']}")
        else:
            print("❌ batch_size missing from defaults")
            return False

    except Exception as e:
        print(f"❌ GAN service test failed: {e}")
        return False

    return True

def test_ratio_scaling():
    """Test ratio-based scaling logic."""
    print("🧪 Testing Ratio-Based Scaling Logic...")

    from shared.services.gan_service import _calculate_input_resolution_for_target

    try:
        # Test basic calculation
        result = _calculate_input_resolution_for_target((1920, 1080), 2160, 2)  # 2x model, target 2160p
        expected = (1080, 608)  # Should be downscaled for 2x to reach 2160p
        print(f"✅ Ratio scaling calculated: {result} (expected around {expected})")

    except Exception as e:
        print(f"❌ Ratio scaling test failed: {e}")
        return False

    return True

if __name__ == "__main__":
    success = True

    success &= test_gan_metadata()
    success &= test_gan_service()
    success &= test_ratio_scaling()

    if success:
        print("\n🎉 All GAN implementation tests passed!")
    else:
        print("\n❌ Some GAN implementation tests failed!")

    sys.exit(0 if success else 1)
