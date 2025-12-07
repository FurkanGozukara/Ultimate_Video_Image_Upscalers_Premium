import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .path_utils import collision_safe_path, normalize_path, ffmpeg_set_fps
from .face_restore import restore_image, restore_video


class RealESRGANResult:
    def __init__(self, returncode: int, output_path: Optional[str], log: str):
        self.returncode = returncode
        self.output_path = output_path
        self.log = log


def _run_realesrgan_image(model_name: str, inp: Path, out_dir: Path, face_enhance: bool, outscale: int, model_path: Optional[str] = None, gpu_id: Optional[int] = None) -> Tuple[int, str]:
    cmd = [
        "python",
        str(Path(__file__).parent.parent / "Real-ESRGAN" / "inference_realesrgan.py"),
        "-i",
        str(inp),
        "-o",
        str(out_dir),
        "-n",
        model_name,
        "-s",
        str(outscale),
    ]
    if model_path:
        cmd.extend(["--model_path", model_path])
    if gpu_id is not None:
        cmd.extend(["-g", str(gpu_id)])
    if face_enhance:
        cmd.append("--face_enhance")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode, proc.stdout + "\n" + proc.stderr


def _run_realesrgan_video(model_name: str, inp: Path, out_dir: Path, face_enhance: bool, outscale: int, model_path: Optional[str] = None, gpu_id: Optional[int] = None) -> Tuple[int, str, Optional[Path]]:
    frames = out_dir / "frames"
    frames_up = out_dir / "frames_up"
    frames.mkdir(parents=True, exist_ok=True)
    frames_up.mkdir(parents=True, exist_ok=True)

    subprocess.run(["ffmpeg", "-y", "-i", str(inp), str(frames / "frame_%05d.png")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    rc, log = _run_realesrgan_image(model_name, frames, frames_up, face_enhance, outscale, model_path=model_path, gpu_id=gpu_id)
    if rc != 0:
        return rc, log, None

    out_path = collision_safe_path(inp.with_name(f"{inp.stem}_realESR.mp4"))
    subprocess.run(
        ["ffmpeg", "-y", "-framerate", "30", "-i", str(frames_up / "frame_%05d_out.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return 0, log, out_path if out_path.exists() else None


def run_realesrgan(settings: Dict[str, Any], apply_face: bool = False) -> RealESRGANResult:
    """
    Run Real-ESRGAN for images or videos using inference_realesrgan.py.
    """
    input_path = Path(normalize_path(settings.get("input_path", "")))
    if not input_path.exists():
        return RealESRGANResult(1, None, "Input missing")
    model_name = settings.get("model") or settings.get("model_name") or "RealESRGAN_x4plus"
    model_name = Path(model_name).stem
    model_path = settings.get("model_path")
    outscale = int(settings.get("scale") or 4)
    fps_override = float(settings.get("fps_override") or 0)
    gpu_id = None
    if settings.get("cuda_device"):
        try:
            first = str(settings["cuda_device"]).split(",")[0].strip()
            if first.isdigit():
                gpu_id = int(first)
        except Exception:
            gpu_id = None

    work = Path(tempfile.mkdtemp(prefix="realesrgan_"))
    try:
        if input_path.suffix.lower() in (".mp4", ".mov", ".mkv", ".avi"):
            rc, log, out_path = _run_realesrgan_video(model_name, input_path, work, apply_face, outscale, model_path=model_path, gpu_id=gpu_id)
            if rc != 0 or not out_path:
                return RealESRGANResult(rc, None, log)
            if fps_override:
                out_path = ffmpeg_set_fps(out_path, fps_override)
            if apply_face and out_path and out_path.exists():
                restored = restore_video(out_path, strength=0.5, on_progress=None)
                if restored:
                    out_path = Path(restored)
            return RealESRGANResult(0, str(out_path), log)
        else:
            rc, log = _run_realesrgan_image(model_name, input_path, work, apply_face, outscale, model_path=model_path, gpu_id=gpu_id)
            if rc != 0:
                return RealESRGANResult(rc, None, log)
            candidates = sorted(work.glob("*_out.*"))
            if not candidates:
                return RealESRGANResult(1, None, "No outputs produced")
            out_path = collision_safe_path(input_path.with_name(f"{input_path.stem}_realESR{candidates[0].suffix}"))
            shutil.copyfile(candidates[0], out_path)
            if apply_face and out_path.exists():
                restored = restore_image(str(out_path), strength=0.5)
                if restored:
                    out_path = Path(restored)
            return RealESRGANResult(0, str(out_path), log)
    finally:
        shutil.rmtree(work, ignore_errors=True)

