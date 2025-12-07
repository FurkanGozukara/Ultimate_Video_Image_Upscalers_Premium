import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from .path_utils import collision_safe_dir, collision_safe_path, ffmpeg_set_fps, normalize_path, get_media_fps
from .face_restore import restore_image, restore_video


class GanResult:
    def __init__(self, returncode: int, output_path: Optional[str], log: str):
        self.returncode = returncode
        self.output_path = output_path
        self.log = log


def _upscale_image(input_path: Path, scale: int, output_format: str = "auto") -> Path:
    img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Failed to read image")
    h, w = img.shape[:2]
    up = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)
    if output_format == "png":
        out = collision_safe_path(input_path.with_name(f"{input_path.stem}_gan.png"))
    else:
        out = collision_safe_path(input_path.with_name(f"{input_path.stem}_gan{input_path.suffix}"))
    cv2.imwrite(str(out), up)
    return out


def _upscale_video(input_path: Path, scale: int, output_format: str = "auto", fps_override: float = 0) -> Path:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH")

    work = Path(tempfile.mkdtemp(prefix="gan_video_"))
    frames_dir = work / "frames"
    frames_out = work / "frames_out"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frames_out.mkdir(parents=True, exist_ok=True)

    subprocess.run(["ffmpeg", "-y", "-i", str(input_path), str(frames_dir / "frame_%05d.png")],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for frame in sorted(frames_dir.glob("frame_*.png")):
        img = cv2.imread(str(frame), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        h, w = img.shape[:2]
        up = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(str(frames_out / frame.name), up)

    if output_format == "png":
        out_dir = collision_safe_dir(input_path.parent / f"{input_path.stem}_gan")
        shutil.move(str(frames_out), out_dir)
        shutil.rmtree(work, ignore_errors=True)
        return out_dir

    source_fps = get_media_fps(str(input_path)) or 30.0
    use_fps = fps_override if fps_override and fps_override > 0 else source_fps
    out = collision_safe_path(input_path.with_name(f"{input_path.stem}_gan.mp4"))
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(use_fps),
            "-i",
            str(frames_out / "frame_%05d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    shutil.rmtree(work, ignore_errors=True)
    return out


def run_gan_upscale(settings: Dict[str, Any], apply_face: bool = False) -> GanResult:
    """
    Lightweight CPU upscale using OpenCV Lanczos (fallback when Real-ESRGAN not selected).
    """
    input_path = Path(normalize_path(settings.get("input_path", "")))
    if not input_path.exists():
        return GanResult(1, None, "Input missing")
    scale = int(settings.get("scale", 2))
    if scale not in (2, 4):
        scale = 2
    fps_override = float(settings.get("fps_override") or 0)
    output_format = settings.get("output_format") or "auto"
    if output_format not in ("auto", "mp4", "png"):
        output_format = "auto"

    log_lines = []
    try:
        if input_path.suffix.lower() in (".mp4", ".mov", ".mkv", ".avi"):
            out = _upscale_video(input_path, scale, output_format=output_format, fps_override=fps_override)
            if apply_face and out and Path(out).exists() and Path(out).is_file():
                restored = restore_video(out, strength=0.5, on_progress=log_lines.append)
                if restored:
                    out = restored
                    log_lines.append(f"Face-restored video saved to {restored}")
        else:
            out = _upscale_image(input_path, scale, output_format=output_format)
            if apply_face and out and Path(out).exists():
                restored = restore_image(out, strength=0.5)
                if restored:
                    out = restored
                    log_lines.append(f"Face-restored image saved to {restored}")
        return GanResult(0, str(out), "\n".join(log_lines))
    except Exception as exc:
        log_lines.append(str(exc))
        return GanResult(1, None, "\n".join(log_lines))

