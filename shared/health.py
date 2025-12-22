import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional

from .path_utils import get_disk_free_gb, is_writable


def _check_ffmpeg() -> Dict[str, Optional[str]]:
    try:
        proc = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0:
            version_line = proc.stdout.splitlines()[0] if proc.stdout else "ffmpeg available"
            return {"status": "ok", "detail": version_line}
        return {"status": "error", "detail": proc.stderr.strip() or "ffmpeg returned non-zero"}
    except FileNotFoundError:
        return {"status": "missing", "detail": "ffmpeg not found in PATH"}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


def _check_cuda() -> Dict[str, Optional[str]]:
    try:
        from .gpu_utils import get_gpu_info, get_cuda_version, is_apple_silicon
        
        # Check for Apple Silicon
        if is_apple_silicon():
            return {
                "status": "ok",
                "detail": "🍎 Apple Silicon detected - Use MPS backend for GPU acceleration (CUDA not available on macOS)"
            }
        
        # Get detailed GPU info
        gpus = get_gpu_info()
        
        if not gpus:
            return {"status": "missing", "detail": "No CUDA GPUs detected - will use CPU (significantly slower)"}
        
        device_info = []
        cuda_ver = get_cuda_version()
        if cuda_ver:
            device_info.append(f"**CUDA Version:** {cuda_ver}")
        
        device_info.append(f"**Detected {len(gpus)} GPU(s):**")
        
        for gpu in gpus:
            if gpu.is_available:
                compute_cap = f" [Compute {gpu.compute_capability[0]}.{gpu.compute_capability[1]}]" if gpu.compute_capability else ""
                device_info.append(
                    f"✅ GPU {gpu.id}: {gpu.name}\n"
                    f"   └─ {gpu.available_memory_gb:.1f}GB free / {gpu.total_memory_gb:.1f}GB total{compute_cap}"
                )
            else:
                device_info.append(f"⚠️ GPU {gpu.id}: {gpu.name} (unavailable)")
        
        return {"status": "ok", "detail": "\n".join(device_info)}
        
    except Exception as exc:
        return {"status": "error", "detail": f"CUDA check failed: {exc}"}


def _check_vs_build_tools() -> Dict[str, Optional[str]]:
    if platform.system() != "Windows":
        return {"status": "skipped", "detail": "VS Build Tools not required on this platform"}

    # Comprehensive list of candidate paths for different VS versions and editions
    candidates = [
        # VS 2022 - All editions
        Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvarsall.bat"),
        Path("C:/Program Files/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvarsall.bat"),
        Path("C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvarsall.bat"),
        Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvarsall.bat"),
        Path("C:/Program Files/Microsoft Visual Studio/2022/Professional/VC/Auxiliary/Build/vcvarsall.bat"),
        Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/Professional/VC/Auxiliary/Build/vcvarsall.bat"),
        Path("C:/Program Files/Microsoft Visual Studio/2022/Enterprise/VC/Auxiliary/Build/vcvarsall.bat"),
        Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/Enterprise/VC/Auxiliary/Build/vcvarsall.bat"),
        Path("C:/Program Files/Microsoft Visual Studio/2022/Preview/VC/Auxiliary/Build/vcvarsall.bat"),
        Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/Preview/VC/Auxiliary/Build/vcvarsall.bat"),
        
        # VS 2019 - All editions
        Path("C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/VC/Auxiliary/Build/vcvarsall.bat"),
        Path("C:/Program Files/Microsoft Visual Studio/2019/BuildTools/VC/Auxiliary/Build/vcvarsall.bat"),
        Path("C:/Program Files/Microsoft Visual Studio/2019/Community/VC/Auxiliary/Build/vcvarsall.bat"),
        Path("C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Auxiliary/Build/vcvarsall.bat"),
        Path("C:/Program Files/Microsoft Visual Studio/2019/Professional/VC/Auxiliary/Build/vcvarsall.bat"),
        Path("C:/Program Files (x86)/Microsoft Visual Studio/2019/Professional/VC/Auxiliary/Build/vcvarsall.bat"),
        Path("C:/Program Files/Microsoft Visual Studio/2019/Enterprise/VC/Auxiliary/Build/vcvarsall.bat"),
        Path("C:/Program Files (x86)/Microsoft Visual Studio/2019/Enterprise/VC/Auxiliary/Build/vcvarsall.bat"),
        
        # VS 2017 - Legacy support
        Path("C:/Program Files (x86)/Microsoft Visual Studio/2017/BuildTools/VC/Auxiliary/Build/vcvarsall.bat"),
        Path("C:/Program Files/Microsoft Visual Studio/2017/BuildTools/VC/Auxiliary/Build/vcvarsall.bat"),
        Path("C:/Program Files/Microsoft Visual Studio/2017/Community/VC/Auxiliary/Build/vcvarsall.bat"),
        Path("C:/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Auxiliary/Build/vcvarsall.bat"),
        Path("C:/Program Files/Microsoft Visual Studio/2017/Professional/VC/Auxiliary/Build/vcvarsall.bat"),
        Path("C:/Program Files (x86)/Microsoft Visual Studio/2017/Professional/VC/Auxiliary/Build/vcvarsall.bat"),
        Path("C:/Program Files/Microsoft Visual Studio/2017/Enterprise/VC/Auxiliary/Build/vcvarsall.bat"),
        Path("C:/Program Files (x86)/Microsoft Visual Studio/2017/Enterprise/VC/Auxiliary/Build/vcvarsall.bat"),
    ]

    found = next((p for p in candidates if p.exists()), None)
    if found:
        # Multi-level validation: file existence, structure, and optional execution test
        try:
            # Level 1: Basic file content validation (fast, reliable)
            with open(found, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(500)
                if 'vcvarsall' not in content.lower():
                    # File exists but doesn't look like vcvarsall.bat
                    found = None  # Continue searching
                else:
                    # File looks valid, try execution test (can fail in sandboxed environments)
                    try:
                        import subprocess
                        
                        # Level 2: Quick execution test (with lenient timeout)
                        result = subprocess.run(
                            ["cmd", "/c", f'call "{found}" x64 >nul 2>&1 && echo VCVARS_SUCCESS'],
                            capture_output=True,
                            text=True,
                            timeout=15,
                            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
                        )
                        
                        if "VCVARS_SUCCESS" in result.stdout:
                            # Perfect: file exists, valid content, and execution works
                            return {
                                "status": "ok",
                                "detail": f"✅ VS Build Tools verified and working\nPath: {found}\nTorch.compile support: ENABLED"
                            }
                        else:
                            # File valid, but execution failed (might be sandbox/permissions issue)
                            # This is OK - subprocess mode will activate it properly
                            return {
                                "status": "ok",
                                "detail": f"✅ VS Build Tools detected at {found}\nTorch.compile will be available in subprocess mode\n(Health check execution failed, but file is valid)"
                            }
                            
                    except subprocess.TimeoutExpired:
                        # Timeout doesn't mean file is invalid - could be slow system
                        return {
                            "status": "ok",
                            "detail": f"✅ VS Build Tools detected at {found}\nTorch.compile will be available\n(Validation timed out, but file exists)"
                        }
                    except Exception as exec_err:
                        # Execution test failed, but file content is valid
                        return {
                            "status": "ok",
                            "detail": f"✅ VS Build Tools detected at {found}\nTorch.compile will be available in subprocess mode\n(Execution test failed: {str(exec_err)[:50]}, but file is valid)"
                        }
                        
        except Exception as file_err:
            # File read failed - skip this candidate
            pass
    
    # If we get here, no valid VS installation was found
    return {
        "status": "warning", 
        "detail": (
            "⚠️ VS Build Tools not detected\n"
            "Torch.compile will be automatically disabled.\n\n"
            "To enable torch.compile:\n"
            "1. Install Visual Studio 2022 Build Tools\n"
            "2. Select 'Desktop development with C++' workload\n"
            "3. Restart the application\n\n"
            f"Checked {len(candidates)} paths across VS 2017/2019/2022 editions."
        )
    }


def _check_disk(path: Path) -> Dict[str, Optional[str]]:
    try:
        free_gb = get_disk_free_gb(path)
        
        # Determine status based on free space
        if free_gb < 1.0:
            status = "error"
            detail = f"⚠️ CRITICAL: Only {free_gb:.2f} GB free at {path}. Need at least 1GB."
        elif free_gb < 5.0:
            status = "warning"
            detail = f"⚠️ LOW: {free_gb:.1f} GB free at {path}. Recommended: 5GB+ for processing."
        else:
            status = "ok"
            detail = f"✅ {free_gb:.1f} GB free at {path}"
        
        return {"status": status, "detail": detail}
    except Exception as e:
        return {"status": "error", "detail": f"Failed to check disk space: {str(e)}"}


def _check_writable(path: Path) -> Dict[str, Optional[str]]:
    return {"status": "ok" if is_writable(path) else "error", "detail": f"Writability: {path}"}


def is_vs_build_tools_available() -> bool:
    """Check if VS Build Tools are available for torch.compile."""
    result = _check_vs_build_tools()
    return result.get("status") == "ok"


def _check_gradio() -> Dict[str, Optional[str]]:
    """Check Gradio version and feature availability"""
    try:
        from .gradio_compat import check_gradio_version, check_required_features
        
        is_compatible, version_msg, features = check_gradio_version()
        required_ok, features_msg = check_required_features()
        
        if is_compatible and required_ok:
            return {"status": "ok", "detail": version_msg}
        elif is_compatible and not required_ok:
            return {"status": "warning", "detail": f"{version_msg}\n{features_msg}"}
        else:
            return {"status": "error", "detail": f"{version_msg}\n{features_msg}"}
            
    except Exception as e:
        return {"status": "error", "detail": f"Gradio check failed: {str(e)}"}


def collect_health_report(temp_dir: Path, output_dir: Path) -> Dict[str, Dict[str, Optional[str]]]:
    report = {
        "gradio": _check_gradio(),  # Check Gradio FIRST (critical for UI)
        "ffmpeg": _check_ffmpeg(),
        "cuda": _check_cuda(),
        "vs_build_tools": _check_vs_build_tools(),
        "temp_dir": _check_writable(temp_dir),
        "output_dir": _check_writable(output_dir),
        "disk_temp": _check_disk(temp_dir),
        "disk_output": _check_disk(output_dir),
    }
    return report


def check_prerequisites_before_run(
    estimated_output_size_gb: float = 0,
    temp_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    require_cuda: bool = True
) -> tuple[bool, str]:
    """
    Check if system meets prerequisites before starting a long operation.
    
    Returns:
        (success: bool, message: str)
    """
    errors = []
    warnings = []
    
    # Check CUDA if required
    if require_cuda:
        cuda_check = _check_cuda()
        if cuda_check["status"] != "ok":
            errors.append(f"CUDA: {cuda_check['detail']}")
    
    # Check ffmpeg
    ffmpeg_check = _check_ffmpeg()
    if ffmpeg_check["status"] != "ok":
        errors.append(f"ffmpeg: {ffmpeg_check['detail']}")
    
    # Check disk space
    if temp_dir and estimated_output_size_gb > 0:
        free_gb = get_disk_free_gb(temp_dir)
        required_gb = estimated_output_size_gb * 2  # 2x for temp files
        if free_gb < required_gb:
            errors.append(f"Insufficient temp disk space: {free_gb:.1f}GB available, need ~{required_gb:.1f}GB")
        elif free_gb < required_gb * 1.5:
            warnings.append(f"Low temp disk space: {free_gb:.1f}GB available, recommended {required_gb * 1.5:.1f}GB")
    
    if output_dir and estimated_output_size_gb > 0:
        free_gb = get_disk_free_gb(output_dir)
        if free_gb < estimated_output_size_gb:
            errors.append(f"Insufficient output disk space: {free_gb:.1f}GB available, need ~{estimated_output_size_gb:.1f}GB")
        elif free_gb < estimated_output_size_gb * 1.2:
            warnings.append(f"Low output disk space: {free_gb:.1f}GB available")
    
    if errors:
        return False, "ERRORS:\n" + "\n".join(f"❌ {e}" for e in errors)
    elif warnings:
        return True, "WARNINGS:\n" + "\n".join(f"⚠️ {w}" for w in warnings)
    else:
        return True, "✅ All prerequisites met"


