import shutil
import subprocess
import tempfile
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .path_utils import (
    collision_safe_dir,
    collision_safe_path,
    ffmpeg_set_fps,
    normalize_path,
    get_media_fps,
    resolve_output_location,
    detect_input_type,
)
from .face_restore import restore_image, restore_video


# GAN Model Metadata System
@dataclass
class GanModelMetadata:
    name: str
    scale: int
    architecture: str
    input_channels: int = 3
    output_channels: int = 3
    description: str = ""
    author: str = "unknown"
    tags: list = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


# Real-ESRGAN model definitions with scale factors
REAL_ESRGAN_MODELS = {
    'RealESRGAN_x4plus': {'scale': 4, 'arch': 'RRDBNet'},
    'RealESRNet_x4plus': {'scale': 4, 'arch': 'RRDBNet'},
    'RealESRGAN_x4plus_anime_6B': {'scale': 4, 'arch': 'RRDBNet'},
    'RealESRGAN_x2plus': {'scale': 2, 'arch': 'RRDBNet'},
    'realesr-animevideov3': {'scale': 4, 'arch': 'SRVGGNetCompact'},
    'realesr-general-x4v3': {'scale': 4, 'arch': 'SRVGGNetCompact'},
}


class GanModelRegistry:
    """Comprehensive registry for GAN model metadata from multiple sources"""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self._omd_cache: Dict[str, GanModelMetadata] = {}
        self._loaded = False

    def _load_omd_metadata(self) -> None:
        """Load metadata from Open Model Database"""
        if self._loaded:
            return

        omd_dir = self.base_dir / "open-model-database" / "data" / "models"
        if not omd_dir.exists():
            self._loaded = True
            return

        for json_file in omd_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                scale = data.get('scale', 4)
                name = data.get('name', json_file.stem)
                arch = data.get('architecture', 'esrgan')

                metadata = GanModelMetadata(
                    name=name,
                    scale=int(scale),
                    architecture=arch,
                    input_channels=data.get('inputChannels', 3),
                    output_channels=data.get('outputChannels', 3),
                    description=data.get('description', ''),
                    author=data.get('author', 'unknown'),
                    tags=data.get('tags', [])
                )

                # Use normalized key for lookup
                key = self._normalize_name(name)
                self._omd_cache[key] = metadata

            except Exception:
                continue

        self._loaded = True

    def _normalize_name(self, name: str) -> str:
        """Normalize model name for consistent lookup"""
        # Remove scale prefix, file extensions, and normalize
        name = re.sub(r'^\d+x[-_]', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\.(pth|safetensors|onnx)$', '', name, flags=re.IGNORECASE)
        return re.sub(r'[^a-z0-9]', '', name.lower())

    def get_model_metadata(self, model_filename: str) -> GanModelMetadata:
        """Get comprehensive metadata for a model file"""
        # First check Real-ESRGAN hardcoded models
        if model_filename in REAL_ESRGAN_MODELS:
            info = REAL_ESRGAN_MODELS[model_filename]
            return GanModelMetadata(
                name=model_filename,
                scale=info['scale'],
                architecture=info['arch'],
                description=f"Real-ESRGAN {model_filename}"
            )

        # Load OMD data if not already loaded
        self._load_omd_metadata()

        # Try exact filename match in OMD
        if model_filename in self._omd_cache:
            return self._omd_cache[model_filename]

        # Try normalized name match
        normalized = self._normalize_name(model_filename)
        if normalized in self._omd_cache:
            return self._omd_cache[normalized]

        # Fallback: parse scale from filename
        scale = self._parse_scale_from_filename(model_filename)

        return GanModelMetadata(
            name=model_filename,
            scale=scale,
            architecture="unknown",
            description="Unknown GAN model"
        )

    def _parse_scale_from_filename(self, filename: str) -> int:
        """Fallback scale detection from filename patterns"""
        filename_lower = filename.lower()

        # Check for explicit scale prefixes (e.g., "4x_", "2x")
        scale_match = re.search(r'^(\d+)x[-_]', filename_lower)
        if scale_match:
            try:
                return int(scale_match.group(1))
            except ValueError:
                pass

        # Check for scale in middle of filename
        scale_match = re.search(r'(\d+)x', filename_lower)
        if scale_match:
            try:
                scale = int(scale_match.group(1))
                # Only accept reasonable scales (1-16)
                if 1 <= scale <= 16:
                    return scale
            except ValueError:
                pass

        # Default fallback
        return 4


# Global registry instance
_gan_registry: Optional[GanModelRegistry] = None

def get_gan_model_metadata(model_filename: str, base_dir: Path) -> GanModelMetadata:
    """Get metadata for a GAN model"""
    global _gan_registry
    if _gan_registry is None:
        _gan_registry = GanModelRegistry(base_dir)
    return _gan_registry.get_model_metadata(model_filename)


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


def _upscale_video(
    input_path: Path,
    scale: int,
    output_format: str = "auto",
    fps_override: float = 0,
    frames_per_batch: int = 0,
    cancel_event=None,
    log_lines=None,
    png_padding: int = 5,
) -> Path:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH")

    work = Path(tempfile.mkdtemp(prefix="gan_video_"))
    frames_dir = work / "frames"
    frames_out = work / "frames_out"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frames_out.mkdir(parents=True, exist_ok=True)

    pad_val = max(1, int(png_padding or 5))
    frame_glob = f"frame_%0{pad_val}d.png"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(input_path), str(frames_dir / frame_glob)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    batch_counter = 0
    processed = 0
    frames_list = sorted(frames_dir.glob("frame_*.png"))
    total_frames = len(frames_list)
    for frame in frames_list:
        if cancel_event is not None and cancel_event.is_set():
            break
        img = cv2.imread(str(frame), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        h, w = img.shape[:2]
        up = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(str(frames_out / frame.name), up)
        processed += 1
        if frames_per_batch and frames_per_batch > 0:
            batch_counter += 1
            if batch_counter >= frames_per_batch:
                batch_counter = 0
                if log_lines is not None:
                    log_lines.append(
                        f"Processed {processed}/{total_frames} frames (batch size {frames_per_batch})"
                    )

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
            str(frames_out / frame_glob),
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


def run_gan_upscale(
    settings: Dict[str, Any],
    apply_face: bool = False,
    face_strength: float = 0.5,
    global_output_dir: Optional[str] = None,
    cancel_event=None,
) -> GanResult:
    """
    Comprehensive GAN upscaler with dynamic resolution adjustment based on model metadata.
    """
    input_path = Path(normalize_path(settings.get("input_path", "")))
    if not input_path.exists():
        return GanResult(1, None, "Input missing")

    # Get model metadata for accurate scale information
    model_name = settings.get("model", "")
    base_dir = Path(settings.get("base_dir", Path(__file__).parents[1]))
    model_metadata = get_gan_model_metadata(model_name, base_dir)
    model_scale = model_metadata.scale

    # Apply resolution tab logic for dynamic adjustment
    use_resolution_tab = settings.get("use_resolution_tab", True)
    effective_scale = model_scale

    if use_resolution_tab:
        target_res = settings.get("target_resolution", 1080)
        max_target_res = settings.get("max_target_resolution", 1920)
        auto_resolution = settings.get("auto_resolution", True)
        enable_max_target = settings.get("enable_max_target", True)

        # Get input dimensions for auto-resolution calculation
        from .path_utils import get_media_dimensions
        input_dims = get_media_dimensions(str(input_path))

        if input_dims and auto_resolution:
            w, h = input_dims
            short_side = min(w, h)
            computed_res = min(short_side, target_res or short_side)

            if enable_max_target and max_target_res and max_target_res > 0:
                computed_res = min(computed_res, max_target_res)

            # For GAN models, calculate if we need downscaling before upscaling
            if computed_res < short_side:
                # Input needs to be downscaled to reach target resolution
                downscale_factor = computed_res / short_side
                # Effective scale becomes model_scale * (1/downscale_factor)
                # But we handle this by adjusting the target resolution in the upscale process
                effective_scale = model_scale
                # We'll need to modify the upscale logic to handle this
            else:
                effective_scale = model_scale
        else:
            effective_scale = model_scale

    # Ensure scale is reasonable
    if not (1 <= effective_scale <= 16):
        effective_scale = model_scale

    # Override the settings scale with the computed effective scale
    scale = int(effective_scale)
    fps_override = float(settings.get("fps_override") or 0)
    output_format = settings.get("output_format") or "auto"
    if output_format not in ("auto", "mp4", "png"):
        output_format = "auto"

    log_lines = []
    try:
        if cancel_event and cancel_event.is_set():
            return GanResult(1, None, "Canceled")
        # Predict target path using shared resolver for consistency
        fmt = "png" if output_format == "png" else "mp4"
        predicted = resolve_output_location(
            input_path=str(input_path),
            output_format=fmt,
            global_output_dir=global_output_dir,
            batch_mode=False,
        )
        predicted_path = Path(predicted)

        frames_per_batch = int(settings.get("frames_per_batch") or 0)
        if input_path.suffix.lower() in (".mp4", ".mov", ".mkv", ".avi"):
            out = _upscale_video(
                input_path,
                scale,
                output_format=output_format,
                fps_override=fps_override,
                frames_per_batch=frames_per_batch,
                cancel_event=cancel_event,
                log_lines=log_lines,
            )
        else:
            out = _upscale_image(input_path, scale, output_format=output_format)

        if cancel_event and cancel_event.is_set():
            return GanResult(1, str(out) if out else None, "Canceled")

        # Move into the predicted location if different
        if out and predicted_path:
            dest = predicted_path if predicted_path.suffix else collision_safe_dir(predicted_path)
            if Path(out).is_dir():
                if dest.exists():
                    dest = collision_safe_dir(dest)
                shutil.move(out, dest)
                out = str(dest)
            else:
                if dest.is_dir():
                    dest = collision_safe_path(dest / Path(out).name)
                else:
                    dest = collision_safe_path(dest)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(out, dest)
                out = str(dest)

        if apply_face and out and Path(out).exists():
            if detect_input_type(str(input_path)) == "video" or Path(out).suffix.lower() in (".mp4", ".mov", ".mkv", ".avi"):
                restored = restore_video(out, strength=face_strength, on_progress=log_lines.append)
                if restored:
                    out = restored
                    log_lines.append(f"Face-restored video saved to {restored} (strength {face_strength})")
            else:
                restored = restore_image(out, strength=face_strength)
                if restored:
                    out = restored
                    log_lines.append(f"Face-restored image saved to {restored} (strength {face_strength})")

        return GanResult(0, str(out), "\n".join(log_lines))
    except Exception as exc:
        log_lines.append(str(exc))
        return GanResult(1, None, "\n".join(log_lines))

