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
    default_path = Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvarsall.bat")
    if default_path.exists():
        return {"status": "ok", "detail": f"Found vcvarsall at {default_path}"}
    return {"status": "warning", "detail": "VS Build Tools not detected at default path"}


def _check_disk(path: Path) -> Dict[str, Optional[str]]:
    free_gb = get_disk_free_gb(path)
    return {"status": "ok", "detail": f"Free space: {free_gb} GB at {path}"}


def _check_writable(path: Path) -> Dict[str, Optional[str]]:
    return {"status": "ok" if is_writable(path) else "error", "detail": f"Writability: {path}"}


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


