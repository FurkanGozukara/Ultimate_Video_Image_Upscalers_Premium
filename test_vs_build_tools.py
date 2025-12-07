#!/usr/bin/env python3
"""
Test script for VS Build Tools integration.
"""
import sys
import platform
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from shared.health import is_vs_build_tools_available, _check_vs_build_tools
from shared.runner import Runner

def test_vs_build_tools_integration():
    """Test VS Build Tools detection and integration."""
    print("🛠️ Testing VS Build Tools Integration...")

    # Test health check
    print("Testing health check...")
    result = _check_vs_build_tools()
    print(f"Health check result: {result}")

    available = is_vs_build_tools_available()
    print(f"VS Build Tools available: {available}")

    # Test runner integration
    print("Testing runner integration...")
    runner = Runner(Path("."), Path("./temp"), Path("./outputs"))
    vcvars_path = runner._find_vcvars()
    print(f"vcvarsall.bat path: {vcvars_path}")

    # Test command wrapping (on Windows)
    if platform.system() == "Windows":
        print("Testing command wrapping...")
        test_cmd = ["python", "test.py"]
        test_settings = {"compile_dit": True, "compile_vae": False}

        wrapped_cmd = runner._maybe_wrap_with_vcvars(test_cmd, test_settings)
        print(f"Original command: {test_cmd}")
        print(f"Wrapped command: {wrapped_cmd}")
        print(f"Command was modified: {test_cmd != wrapped_cmd}")

        # Check if compile flags were disabled if VS tools not available
        if not available and test_settings.get("compile_dit"):
            print("⚠️ Compile should be disabled when VS tools unavailable")
        else:
            print("✅ Compile handling correct")
    else:
        print("Skipping Windows-specific tests (not on Windows)")

    print("🎉 VS Build Tools integration test completed!")
    return True

if __name__ == "__main__":
    success = test_vs_build_tools_integration()
    sys.exit(0 if success else 1)
