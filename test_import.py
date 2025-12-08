#!/usr/bin/env python3
"""
Simple import test for the SECourses Ultimate Upscaler
"""

try:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))

    print("Testing imports...")

    # Test core shared modules
    from shared.preset_manager import PresetManager
    print("✅ PresetManager imported")

    from shared.runner import Runner
    print("✅ Runner imported")

    from shared.services.seedvr2_service import seedvr2_defaults
    print("✅ SeedVR2 service imported")

    from shared.services.rife_service import rife_defaults
    print("✅ RIFE service imported")

    from shared.services.gan_service import gan_defaults
    print("✅ GAN service imported")

    # Test main app import
    import secourses_app
    print("✅ Main app imported successfully")

    print("\n🎉 All imports successful! Application is ready to launch.")

except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
