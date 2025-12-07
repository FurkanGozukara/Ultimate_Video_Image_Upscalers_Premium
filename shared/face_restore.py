import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .path_utils import collision_safe_path, normalize_path


def _has_gfpgan() -> bool:
    return shutil.which("gfpgan") is not None or shutil.which("python") is not None


def _run_gfpgan(input_path: str, output_dir: Path, suffix: str = "restored") -> bool:
    """
    Invoke gfpgan CLI to restore faces. Assumes gfpgan is installed and in PATH.
    """
    # Try python -m gfpgan invocation
    cmd = [
        "python",
        "-m",
        "gfpgan",
        "--input",
        input_path,
        "--output",
        str(output_dir),
        "--suffix",
        suffix,
        "--bg_upsampler",
        "none",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode == 0


def restore_image(image_path: str, strength: float = 0.5) -> Optional[str]:
    if not _has_gfpgan():
        return None
    img = Path(normalize_path(image_path))
    out_dir = img.parent / "face_restore"
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = _run_gfpgan(str(img), out_dir)
    if not ok:
        return None
    # gfpgan appends suffix; pick latest
    candidates = sorted(out_dir.glob(f"{img.stem}*"), key=os.path.getmtime)
    if not candidates:
        return None
    final = collision_safe_path(out_dir / f"{img.stem}_fr{img.suffix}")

    # Blend with original based on strength (0 = original, 1 = restored)
    try:
        strength = float(strength)
    except Exception:
        strength = 0.5
    strength = max(0.0, min(1.0, strength))

    orig = cv2.imread(str(img), cv2.IMREAD_UNCHANGED)
    restored = cv2.imread(str(candidates[-1]), cv2.IMREAD_UNCHANGED)
    if orig is not None and restored is not None and 0.0 <= strength < 1.0:
        blended = cv2.addWeighted(restored.astype(np.float32), strength, orig.astype(np.float32), 1.0 - strength, 0)
        cv2.imwrite(str(final), blended.astype(orig.dtype))
    else:
        shutil.copyfile(candidates[-1], final)
    return str(final)


def restore_video(video_path: str, strength: float = 0.5, on_progress: Optional[Callable[[str], None]] = None) -> Optional[str]:
    if not _has_gfpgan():
        if on_progress:
            on_progress("gfpgan not available; skipping face restoration\n")
        return None
    video = Path(normalize_path(video_path))
    work = Path(tempfile.mkdtemp(prefix="face_restore_"))
    frames_dir = work / "frames"
    restored_dir = work / "restored"
    frames_dir.mkdir(parents=True, exist_ok=True)
    restored_dir.mkdir(parents=True, exist_ok=True)

    # Extract frames
    extract_cmd = ["ffmpeg", "-y", "-i", str(video), str(frames_dir / "frame_%05d.png")]
    subprocess.run(extract_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    ok = _run_gfpgan(str(frames_dir), restored_dir)
    if not ok:
        if on_progress:
            on_progress("gfpgan restoration failed\n")
        shutil.rmtree(work, ignore_errors=True)
        return None

    # Blend restored frames with originals based on strength
    try:
        strength_val = float(strength)
    except Exception:
        strength_val = 0.5
    strength_val = max(0.0, min(1.0, strength_val))
    if strength_val < 1.0:
        for frame_file in sorted(restored_dir.glob("frame_*.png")):
            orig_file = frames_dir / frame_file.name
            if not orig_file.exists():
                continue
            orig = cv2.imread(str(orig_file), cv2.IMREAD_UNCHANGED)
            restored = cv2.imread(str(frame_file), cv2.IMREAD_UNCHANGED)
            if orig is None or restored is None:
                continue
            blended = cv2.addWeighted(restored.astype(np.float32), strength_val, orig.astype(np.float32), 1.0 - strength_val, 0)
            cv2.imwrite(str(frame_file), blended.astype(orig.dtype))

    # Re-encode video (no audio)
    out_path = collision_safe_path(video.parent / f"{video.stem}_fr.mp4")
    encode_cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        "30",
        "-i",
        str(restored_dir / "frame_%05d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    subprocess.run(encode_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shutil.rmtree(work, ignore_errors=True)
    if out_path.exists():
        return str(out_path)
    return None

