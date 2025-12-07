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
        import torch

        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            names = [torch.cuda.get_device_name(i) for i in range(count)]
            return {"status": "ok", "detail": f"{count} CUDA device(s): {', '.join(names)}"}
        return {"status": "missing", "detail": "CUDA not available"}
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
        # Test if the vcvarsall.bat file is actually executable
        try:
            # Quick test by checking if we can read the file
            with open(found, 'r') as f:
                if 'vcvarsall.bat' in f.read(200):
                    return {"status": "ok", "detail": f"Found vcvarsall at {found}"}
        except Exception:
            pass

    return {"status": "warning", "detail": "VS Build Tools not detected; torch.compile will be automatically disabled. Install 'Desktop development with C++' workload in Visual Studio 2022 Build Tools for optimal performance."}


def _check_disk(path: Path) -> Dict[str, Optional[str]]:
    free_gb = get_disk_free_gb(path)
    return {"status": "ok", "detail": f"Free space: {free_gb} GB at {path}"}


def _check_writable(path: Path) -> Dict[str, Optional[str]]:
    return {"status": "ok" if is_writable(path) else "error", "detail": f"Writability: {path}"}


def is_vs_build_tools_available() -> bool:
    """Check if VS Build Tools are available for torch.compile."""
    result = _check_vs_build_tools()
    return result.get("status") == "ok"


def collect_health_report(temp_dir: Path, output_dir: Path) -> Dict[str, Dict[str, Optional[str]]]:
    report = {
        "ffmpeg": _check_ffmpeg(),
        "cuda": _check_cuda(),
        "vs_build_tools": _check_vs_build_tools(),
        "temp_dir": _check_writable(temp_dir),
        "output_dir": _check_writable(output_dir),
        "disk_temp": _check_disk(temp_dir),
        "disk_output": _check_disk(output_dir),
    }
    return report


