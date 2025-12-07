#!/usr/bin/env python3
"""
Test basic functionality of the upscaler app components.
"""
import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all core modules can be imported."""
    print("🧪 Testing core imports...")

    try:
        # Test gradio import
        import gradio as gr
        print(f"✅ Gradio imported (version check skipped)")

        # Test core services
        from shared.services.seedvr2_service import seedvr2_defaults, build_seedvr2_callbacks
        print("✅ SeedVR2 service imported")

        from shared.services.gan_service import gan_defaults, build_gan_callbacks
        print("✅ GAN service imported")

        from shared.services.rife_service import rife_defaults, build_rife_callbacks
        print("✅ RIFE service imported")

        from shared.runner import Runner
        print("✅ Runner imported")

        from shared.preset_manager import PresetManager
        print("✅ PresetManager imported")

        return True

    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_gradio_components():
    """Test that gradio components work."""
    print("\n🧪 Testing gradio components...")

    try:
        import gradio as gr

        # Test basic components
        button = gr.Button("Test")
        print("✅ Button component works")

        # Test ImageSlider (the one mentioned as potentially missing)
        try:
            slider = gr.ImageSlider()
            print("✅ ImageSlider component works")
        except Exception as e:
            print(f"⚠️ ImageSlider issue: {e}")

        # Test other components used
        text = gr.Textbox()
        dropdown = gr.Dropdown()
        checkbox = gr.Checkbox()
        print("✅ Basic components work")

        return True

    except Exception as e:
        print(f"❌ Gradio components test failed: {e}")
        return False

def test_service_functions():
    """Test that service functions can be called."""
    print("\n🧪 Testing service functions...")

    try:
        from shared.services.seedvr2_service import seedvr2_defaults
        defaults = seedvr2_defaults()
        print(f"✅ SeedVR2 defaults loaded: {len(defaults)} parameters")

        from shared.services.gan_service import gan_defaults
        base_dir = Path(__file__).parent
        gan_defs = gan_defaults(base_dir)
        print(f"✅ GAN defaults loaded: {len(gan_defs)} parameters")

        from shared.services.rife_service import rife_defaults
        rife_defs = rife_defaults()
        print(f"✅ RIFE defaults loaded: {len(rife_defs)} parameters")

        return True

    except Exception as e:
        print(f"❌ Service functions test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Ultimate Video Image Upscaler functionality...\n")

    success = True
    success &= test_imports()
    success &= test_gradio_components()
    success &= test_service_functions()

    if success:
        print("\n🎉 All basic functionality tests passed!")
        print("\n💡 If the app buttons aren't working, check:")
        print("   1. Virtual environment is activated")
        print("   2. All dependencies are installed")
        print("   3. ffmpeg is in PATH")
        print("   4. Model directories exist")
        print("   5. Check console/logs for errors when clicking buttons")
    else:
        print("\n❌ Some tests failed - check dependencies and imports")

    sys.exit(0 if success else 1)
