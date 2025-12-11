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

    # Extended list of candidate paths for different VS versions
    candidates = [
        # VS 2022 Build Tools
        Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvarsall.bat"),
        Path("C:/Program Files/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvarsall.bat"),
        # VS 2022 Community/Professional/Enterprise
        Path("C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvarsall.bat"),
        Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvarsall.bat"),
        Path("C:/Program Files/Microsoft Visual Studio/2022/Professional/VC/Auxiliary/Build/vcvarsall.bat"),
        Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/Professional/VC/Auxiliary/Build/vcvarsall.bat"),
        Path("C:/Program Files/Microsoft Visual Studio/2022/Enterprise/VC/Auxiliary/Build/vcvarsall.bat"),
        Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/Enterprise/VC/Auxiliary/Build/vcvarsall.bat"),
        # VS 2019 Build Tools (fallback)
        Path("C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/VC/Auxiliary/Build/vcvarsall.bat"),
        Path("C:/Program Files/Microsoft Visual Studio/2019/BuildTools/VC/Auxiliary/Build/vcvarsall.bat"),
        # VS 2019 Community/Professional/Enterprise
        Path("C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Auxiliary/Build/vcvarsall.bat"),
        Path("C:/Program Files/Microsoft Visual Studio/2019/Professional/VC/Auxiliary/Build/vcvarsall.bat"),
        Path("C:/Program Files/Microsoft Visual Studio/2019/Enterprise/VC/Auxiliary/Build/vcvarsall.bat"),
    ]

    found = next((p for p in candidates if p.exists()), None)
    if found:
        # Test if the vcvarsall.bat file is actually executable and working
        try:
            import subprocess
            
            # Actually test execution with a timeout
            result = subprocess.run(
                ["cmd", "/c", f'call "{found}" x64 >nul 2>&1 && echo VCVARS_SUCCESS'],
                capture_output=True,
                text=True,
                timeout=15  # vcvarsall can take 5-10 seconds
            )
            
            if "VCVARS_SUCCESS" in result.stdout:
                return {
                    "status": "ok",
                    "detail": f"✅ VS Build Tools verified and working\nPath: {found}\nTorch.compile support: ENABLED"
                }
            else:
                # File exists but execution failed
                return {
                    "status": "warning",
                    "detail": f"⚠️ vcvarsall.bat found at {found} but execution failed.\nTorch.compile may not work. Try reinstalling VS Build Tools."
                }
                
        except subprocess.TimeoutExpired:
            return {
                "status": "warning",
                "detail": f"⚠️ vcvarsall.bat found at {found} but validation timed out.\nFile may work, but verification failed."
            }
        except Exception as e:
            # Fallback to simple file check if execution test fails
            try:
                with open(found, 'r', encoding='utf-8', errors='ignore') as f:
                    if 'vcvarsall' in f.read(200).lower():
                        return {
                            "status": "ok",
                            "detail": f"Found vcvarsall at {found}\n(Execution test failed: {str(e)}, but file appears valid)"
                        }
            except Exception:
                pass

    return {"status": "warning", "detail": "VS Build Tools not detected; torch.compile will be automatically disabled. Install 'Desktop development with C++' workload in Visual Studio 2022 Build Tools for optimal performance."}


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


