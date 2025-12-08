import shutil
import subprocess
import tempfile
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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


class GanRunner:
    """
    Runner class for GAN-based upscaling operations.
    Provides a clean interface for the runner system.
    """

    def run_gan_processing(
        self,
        input_path: str,
        model_name: str,
        output_path: str,
        settings: Dict[str, Any],
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> GanResult:
        """
        Run GAN processing with the given settings.
        """
        # Prepare settings for the run_gan_upscale function
        gan_settings = settings.copy()
        gan_settings["input_path"] = input_path
        gan_settings["model"] = model_name
        gan_settings["output_path"] = output_path

        # Add base_dir if not present
        if "base_dir" not in gan_settings:
            gan_settings["base_dir"] = str(Path(__file__).parents[1])

        # Create a progress callback wrapper
        def progress_callback(msg: str):
            if on_progress:
                on_progress(msg)

        # For now, use a simple approach - call the existing run_gan_upscale function
        # TODO: Integrate with threading and cancellation
        try:
            result = run_gan_upscale(
                settings=gan_settings,
                apply_face=settings.get("face_restore_global", False),
                face_strength=settings.get("face_strength", 0.5),
                global_output_dir=str(Path(output_path).parent) if output_path else None,
                cancel_event=None,  # TODO: Add cancellation support
            )

            # Call progress callback if provided
            if on_progress and result.log:
                for line in result.log.split('\n'):
                    if line.strip():
                        progress_callback(line + '\n')

            return result

        except Exception as e:
            error_msg = f"GAN processing failed: {str(e)}"
            if on_progress:
                progress_callback(f"❌ {error_msg}\n")
            return GanResult(1, None, error_msg)


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


def _process_frame_folder(
    frame_dir: Path,
    scale: int,
    output_format: str = "auto",
    cancel_event=None,
    log_lines=None,
) -> Path:
    """Process a folder containing image frames"""
    if not frame_dir.exists() or not frame_dir.is_dir():
        raise ValueError(f"Frame directory not found: {frame_dir}")

    # Find all image frames
    frame_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
    frames = sorted([f for f in frame_dir.iterdir() if f.is_file() and f.suffix.lower() in frame_extensions])

    if not frames:
        raise ValueError(f"No image frames found in {frame_dir}")

    # Create output directory
    output_dir = frame_dir.parent / f"{frame_dir.name}_gan"
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for frame_path in frames:
        if cancel_event and cancel_event.is_set():
            break

        # Process individual frame
        img = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        h, w = img.shape[:2]
        upscaled = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)

        # Save processed frame
        output_path = output_dir / frame_path.name
        cv2.imwrite(str(output_path), upscaled)
        processed += 1

        if log_lines is not None:
            log_lines.append(f"Processed frame {processed}/{len(frames)}: {frame_path.name}")

    if processed == 0:
        raise RuntimeError("No frames were successfully processed")

    return output_dir


def _process_video_frames(
    video_path: Path,
    settings: Dict[str, Any],
    cancel_event=None,
    log_lines: List[str] = None
) -> Path:
    """
    Process video by extracting frames, upscaling each frame in batches, then reassembling.
    """
    if log_lines is None:
        log_lines = []

    import tempfile
    import shutil
    from pathlib import Path

    temp_dir = Path(tempfile.mkdtemp(prefix="gan_video_"))
    try:
        # Extract frames from video
        frames_dir = temp_dir / "frames"
        frames_dir.mkdir()

        log_lines.append(f"Extracting frames from video: {video_path.name}")
        extract_result = subprocess.run([
            "ffmpeg", "-i", str(video_path),
            "-q:v", "2",  # High quality
            str(frames_dir / "frame_%08d.png")
        ], capture_output=True, text=True)

        if extract_result.returncode != 0:
            raise Exception(f"Frame extraction failed: {extract_result.stderr}")

        # Get frame count
        frame_files = sorted(frames_dir.glob("frame_*.png"))
        if not frame_files:
            raise Exception("No frames extracted from video")

        log_lines.append(f"Extracted {len(frame_files)} frames")

        # Get original video FPS for reassembly
        fps_result = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path)
        ], capture_output=True, text=True)

        original_fps = 30.0  # Default
        if fps_result.returncode == 0 and fps_result.stdout.strip():
            rate_str = fps_result.stdout.strip()
            if "/" in rate_str:
                num, den = rate_str.split("/")
                try:
                    original_fps = float(num) / float(den)
                except:
                    pass

        # Process frames in batches
        batch_size = settings.get("batch_size", 1)
        upscaled_frames_dir = temp_dir / "upscaled_frames"
        upscaled_frames_dir.mkdir()

        log_lines.append(f"Processing frames in batches of {batch_size}")

        for i in range(0, len(frame_files), batch_size):
            if cancel_event and cancel_event.is_set():
                raise Exception("Processing cancelled")

            batch_files = frame_files[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(frame_files) + batch_size - 1) // batch_size

            log_lines.append(f"Processing batch {batch_num}/{total_batches} ({len(batch_files)} frames)")

            # Create a temporary frame folder for this batch
            batch_dir = temp_dir / f"batch_{batch_num}"
            batch_dir.mkdir()

            # Copy batch files to temporary directory
            for frame_file in batch_files:
                shutil.copy2(frame_file, batch_dir / frame_file.name)

            # Process this batch as a frame folder
            batch_result = _process_frame_folder(
                Path(batch_dir),
                model_scale,
                "png",  # Force PNG for frames
                cancel_event,
                []
            )

            if batch_result:
                # Move upscaled frames to final directory
                for upscaled_file in batch_result.glob("*.png"):
                    # Extract frame number and rename
                    frame_match = re.search(r'frame_(\d+)\.png', upscaled_file.name)
                    if frame_match:
                        frame_num = frame_match.group(1)
                        final_name = f"frame_{frame_num}.png"
                        shutil.move(str(upscaled_file), str(upscaled_frames_dir / final_name))

            # Clean up batch directory
            shutil.rmtree(batch_dir, ignore_errors=True)

        # Reassemble video from upscaled frames
        log_lines.append("Reassembling video from upscaled frames")

        output_video_path = _reassemble_video_from_frames(
            upscaled_frames_dir,
            video_path,
            original_fps,
            settings
        )

        log_lines.append(f"Video processing complete: {output_video_path}")
        return output_video_path

    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass


def _reassemble_video_from_frames(
    frames_dir: Path,
    original_video: Path,
    fps: float,
    settings: Dict[str, Any]
) -> Path:
    """Reassemble video from upscaled frames."""
    from .path_utils import collision_safe_path

    # Determine output path
    output_dir = Path(settings.get("global_output_dir", original_video.parent))
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = original_video.stem
    output_path = collision_safe_path(output_dir / f"{base_name}_gan_upscaled.mp4")

    # Use ffmpeg to reassemble video
    ffmpeg_cmd = [
        "ffmpeg",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%08d.png"),
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",  # High quality
        "-pix_fmt", "yuv420p",
        "-y",  # Overwrite
        str(output_path)
    ]

    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Video reassembly failed: {result.stderr}")

    return output_path


def _preprocess_input_resolution(input_path: Path, target_resolution: int, temp_dir: Path) -> Path:
    """Preprocess input by downscaling to optimal resolution for GAN upscaling."""
    import cv2

    # Create temp file for preprocessed input
    suffix = input_path.suffix
    temp_input = temp_dir / f"preprocessed_input_{input_path.stem}{suffix}"
    temp_input.parent.mkdir(parents=True, exist_ok=True)

    # For images
    if suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp']:
        img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return input_path

        h, w = img.shape[:2]
        short_side = min(w, h)

        if short_side <= target_resolution:
            return input_path  # No downscaling needed

        # Calculate new dimensions maintaining aspect ratio
        if w <= h:
            new_w = target_resolution
            new_h = int(target_resolution * (h / w))
        else:
            new_h = target_resolution
            new_w = int(target_resolution * (w / h))

        # Downscale
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(str(temp_input), resized)
        return temp_input

    # For videos, we'd need ffmpeg downscaling, but for now return original
    # Video downscaling is more complex and might be handled elsewhere
    return input_path


def run_gan_upscale(
    settings: Dict[str, Any],
    apply_face: bool = False,
    face_strength: float = 0.5,
    global_output_dir: Optional[str] = None,
    cancel_event=None,
) -> GanResult:
    """
    Comprehensive GAN upscaler with dynamic resolution adjustment based on model metadata.
    Supports videos, images, and frame folders.
    """
    input_path = Path(normalize_path(settings.get("input_path", "")))
    if not input_path.exists():
        return GanResult(1, None, "Input missing")

    # Detect input type: video, image, or frame folder
    from .path_utils import detect_input_type
    input_type = detect_input_type(str(input_path))

    # Handle frame folders
    if input_type == "directory":
        log_lines = []
        try:
            output_path = _process_frame_folder(
                input_path,
                settings,
                cancel_event,
                log_lines
            )
            return GanResult(0, str(output_path), "\n".join(log_lines))
        except Exception as e:
            return GanResult(1, None, f"Frame folder processing failed: {e}")

    # Handle video files - extract frames, process, reassemble
    if input_type == "video":
        log_lines = []
        try:
            output_path = _process_video_frames(
                input_path,
                settings,
                cancel_event,
                log_lines
            )
            return GanResult(0, str(output_path), "\n".join(log_lines))
        except Exception as e:
            return GanResult(1, None, f"Video processing failed: {e}")

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

            # For fixed-scale GAN models, calculate optimal input resolution
            if model_scale > 1:
                # Calculate what input resolution would give us target output after scaling
                optimal_input = target_res // model_scale

                # Apply max resolution constraint
                if enable_max_target and max_target_res > 0:
                    max_input = max_target_res // model_scale
                    optimal_input = min(optimal_input, max_input)

                # Don't exceed original resolution
                optimal_input = min(optimal_input, short_side)

                # For ratio-based scaling, we may need to downscale first
                downscale_first = settings.get("downscale_first", True)
                if downscale_first and optimal_input < short_side:
                    # Mark for preprocessing downscale
                    settings["needs_input_downscale"] = True
                    settings["target_input_resolution"] = optimal_input
                    computed_res = optimal_input
                else:
                    # Use optimal input size for direct scaling
                    computed_res = optimal_input
            else:
                # For scale=1 models or when not using ratio logic
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
            input_path=str(processed_input_path),
            output_format=fmt,
            global_output_dir=global_output_dir,
            batch_mode=False,
        )
        predicted_path = Path(predicted)

        # Apply input preprocessing if needed for ratio-based scaling
        processed_input_path = input_path
        if settings.get("needs_input_downscale"):
            target_input_res = settings.get("target_input_resolution", computed_res)
            log_lines.append(f"Preprocessing input: downscaling to {target_input_res}px for optimal {scale}x upscaling")
            processed_input_path = _preprocess_input_resolution(input_path, target_input_res, Path(tempfile.gettempdir()))
            if processed_input_path != input_path:
                log_lines.append("Input preprocessing complete")

        frames_per_batch = int(settings.get("frames_per_batch") or 0)
        if processed_input_path.suffix.lower() in (".mp4", ".mov", ".mkv", ".avi"):
            out = _upscale_video(
                processed_input_path,
                scale,
                output_format=output_format,
                fps_override=fps_override,
                frames_per_batch=frames_per_batch,
                cancel_event=cancel_event,
                log_lines=log_lines,
            )
        else:
            out = _upscale_image(processed_input_path, scale, output_format=output_format)

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

