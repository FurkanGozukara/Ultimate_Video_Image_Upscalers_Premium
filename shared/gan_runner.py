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
    supports_multi_gpu: bool = False

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
        self._spandrel_available = self._check_spandrel()

    def _check_spandrel(self) -> bool:
        """Check if spandrel is available for model loading"""
        try:
            import spandrel
            return True
        except ImportError:
            return False

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

    def get_model_metadata_with_spandrel(self, model_path: Path) -> Optional[GanModelMetadata]:
        """Use spandrel to detect model architecture and scale"""
        if not self._spandrel_available:
            return None

        try:
            import spandrel
            model_descriptor = spandrel.ModelLoader().load_from_file(str(model_path))
            
            # Extract metadata from spandrel
            scale = getattr(model_descriptor.model, 'scale', 4)
            if hasattr(scale, '__iter__'):
                scale = scale[0] if len(scale) > 0 else 4
            
            arch_name = model_descriptor.architecture.name if hasattr(model_descriptor, 'architecture') else "unknown"
            
            return GanModelMetadata(
                name=model_path.stem,
                scale=int(scale),
                architecture=arch_name,
                input_channels=getattr(model_descriptor.model, 'in_channels', 3) if hasattr(model_descriptor.model, 'in_channels') else 3,
                output_channels=getattr(model_descriptor.model, 'out_channels', 3) if hasattr(model_descriptor.model, 'out_channels') else 3,
                description=f"Detected via spandrel: {arch_name}",
                supports_multi_gpu=False  # Most GAN models don't support multi-GPU well
            )
        except Exception as e:
            # Spandrel failed, fall back to other methods
            return None

    def get_model_metadata(self, model_filename: str) -> GanModelMetadata:
        """Get comprehensive metadata for a model file"""
        # First check Real-ESRGAN hardcoded models
        model_stem = Path(model_filename).stem
        if model_stem in REAL_ESRGAN_MODELS:
            info = REAL_ESRGAN_MODELS[model_stem]
            return GanModelMetadata(
                name=model_stem,
                scale=info['scale'],
                architecture=info['arch'],
                description=f"Real-ESRGAN {model_stem}",
                supports_multi_gpu=False
            )

        # Try spandrel first for accurate detection
        model_path = self.base_dir / "Image_Upscale_Models" / model_filename
        if model_path.exists():
            spandrel_meta = self.get_model_metadata_with_spandrel(model_path)
            if spandrel_meta:
                return spandrel_meta

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
            description="Unknown GAN model (fallback detection)"
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

    def reload_cache(self):
        """Force reload of metadata cache"""
        self._loaded = False
        self._omd_cache.clear()
        self._load_omd_metadata()


# Global registry instance
_gan_registry: Optional[GanModelRegistry] = None

def get_gan_model_metadata(model_filename: str, base_dir: Path) -> GanModelMetadata:
    """Get metadata for a GAN model"""
    global _gan_registry
    if _gan_registry is None or _gan_registry.base_dir != base_dir:
        _gan_registry = GanModelRegistry(base_dir)
    return _gan_registry.get_model_metadata(model_filename)


def reload_gan_models_cache(base_dir: Path):
    """Reload GAN model cache when models directory changes"""
    global _gan_registry
    if _gan_registry is not None:
        _gan_registry.reload_cache()
    else:
        _gan_registry = GanModelRegistry(base_dir)


class GanResult:
    def __init__(self, returncode: int, output_path: Optional[str], log: str):
        self.returncode = returncode
        self.output_path = output_path
        self.log = log


def run_gan_upscale(
    input_path: str,
    model_name: str,
    settings: Dict[str, Any],
    base_dir: Path,
    temp_dir: Path,
    output_dir: Path,
    on_progress: Optional[Callable[[str], None]] = None,
    cancel_event=None
) -> GanResult:
    """
    Run GAN-based upscaling using either spandrel or Real-ESRGAN.
    
    This is the main entry point for GAN upscaling operations.
    Supports both image and video upscaling with automatic downscaling
    for ratio-based target resolutions.
    """
    try:
        input_path_obj = Path(normalize_path(input_path))
        if not input_path_obj.exists():
            return GanResult(1, None, f"Input file not found: {input_path}")

        # Get model metadata
        metadata = get_gan_model_metadata(model_name, base_dir)
        model_scale = metadata.scale

        if on_progress:
            on_progress(f"Model: {model_name}, Scale: {model_scale}x, Arch: {metadata.architecture}\n")

        # Determine input type
        input_type = detect_input_type(str(input_path_obj))

        if input_type == "image":
            return _run_gan_image(
                input_path_obj, model_name, settings, base_dir, temp_dir,
                output_dir, metadata, on_progress, cancel_event
            )
        elif input_type in ("video", "directory"):
            return _run_gan_video(
                input_path_obj, model_name, settings, base_dir, temp_dir,
                output_dir, metadata, on_progress, cancel_event
            )
        else:
            return GanResult(1, None, f"Unsupported input type: {input_type}")

    except Exception as e:
        return GanResult(1, None, f"GAN upscale error: {str(e)}")


def _run_gan_image(
    input_path: Path,
    model_name: str,
    settings: Dict[str, Any],
    base_dir: Path,
    temp_dir: Path,
    output_dir: Path,
    metadata: GanModelMetadata,
    on_progress: Optional[Callable[[str], None]] = None,
    cancel_event=None
) -> GanResult:
    """Run GAN upscaling on a single image"""
    
    if cancel_event and cancel_event.is_set():
        return GanResult(1, None, "Canceled")

    try:
        # Use spandrel if available, otherwise fall back to Real-ESRGAN
        use_spandrel = _gan_registry and _gan_registry._spandrel_available
        
        if use_spandrel:
            result = _run_with_spandrel_image(
                input_path, model_name, settings, base_dir, metadata,
                on_progress, cancel_event
            )
        else:
            result = _run_with_realesrgan_image(
                input_path, model_name, settings, base_dir, metadata,
                on_progress, cancel_event
            )

        # Apply face restoration if requested
        if settings.get("face_restore") and result.returncode == 0 and result.output_path:
            if on_progress:
                on_progress("Applying face restoration...\n")
            restored = restore_image(result.output_path, strength=settings.get("face_strength", 0.5))
            if restored:
                result.output_path = restored

        return result

    except Exception as e:
        return GanResult(1, None, f"Image upscale error: {str(e)}")


def _run_gan_video(
    input_path: Path,
    model_name: str,
    settings: Dict[str, Any],
    base_dir: Path,
    temp_dir: Path,
    output_dir: Path,
    metadata: GanModelMetadata,
    on_progress: Optional[Callable[[str], None]] = None,
    cancel_event=None
) -> GanResult:
    """Run GAN upscaling on video (frame-by-frame processing)"""
    
    if cancel_event and cancel_event.is_set():
        return GanResult(1, None, "Canceled")

    work_dir = temp_dir / f"gan_{input_path.stem}"
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Extract frames
        frames_dir = work_dir / "frames"
        frames_up_dir = work_dir / "frames_upscaled"
        frames_dir.mkdir(parents=True, exist_ok=True)
        frames_up_dir.mkdir(parents=True, exist_ok=True)

        if on_progress:
            on_progress("Extracting frames...\n")

        extract_cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            str(frames_dir / "frame_%05d.png")
        ]
        subprocess.run(extract_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        if cancel_event and cancel_event.is_set():
            shutil.rmtree(work_dir, ignore_errors=True)
            return GanResult(1, None, "Canceled after frame extraction")

        # Count frames
        frame_files = sorted(frames_dir.glob("*.png"))
        total_frames = len(frame_files)

        if total_frames == 0:
            return GanResult(1, None, "No frames extracted from video")

        if on_progress:
            on_progress(f"Processing {total_frames} frames...\n")

        # Batch process frames
        batch_size = settings.get("batch_size", 1)
        
        for i in range(0, total_frames, batch_size):
            if cancel_event and cancel_event.is_set():
                break

            batch_frames = frame_files[i:i+batch_size]
            
            if on_progress:
                progress_pct = int((i / total_frames) * 100)
                on_progress(f"Progress: {progress_pct}% ({i}/{total_frames} frames)\n")

            # Process each frame in batch
            for frame_path in batch_frames:
                # Create temp settings for this frame
                frame_settings = settings.copy()
                frame_settings["input_path"] = str(frame_path)
                
                result = _run_gan_image(
                    frame_path, model_name, frame_settings, base_dir, temp_dir,
                    output_dir, metadata, None, cancel_event
                )
                
                if result.returncode != 0:
                    if on_progress:
                        on_progress(f"Warning: Frame {frame_path.name} failed\n")

        if cancel_event and cancel_event.is_set():
            # Compile partial results if any frames were processed
            pass

        # Reconstruct video from upscaled frames
        if on_progress:
            on_progress("Reconstructing video from upscaled frames...\n")

        # Get original FPS
        original_fps = get_media_fps(str(input_path)) or 30.0
        fps_override = settings.get("fps_override", 0)
        target_fps = fps_override if fps_override > 0 else original_fps

        output_path = resolve_output_location(
            input_path=str(input_path),
            output_format=settings.get("output_format", "mp4"),
            global_output_dir=str(output_dir),
            batch_mode=False
        )

        encode_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(target_fps),
            "-i", str(frames_up_dir / "frame_%05d_out.png"),
            "-c:v", settings.get("video_codec", "libx264"),
            "-crf", str(settings.get("video_quality", 18)),
            "-pix_fmt", "yuv420p",
            str(output_path)
        ]
        
        subprocess.run(encode_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        # Cleanup temp directory
        if not settings.get("keep_temp", False):
            shutil.rmtree(work_dir, ignore_errors=True)

        # Apply face restoration to video if requested
        if settings.get("face_restore") and Path(output_path).exists():
            if on_progress:
                on_progress("Applying face restoration to video...\n")
            restored = restore_video(output_path, strength=settings.get("face_strength", 0.5), on_progress=on_progress)
            if restored:
                output_path = restored

        return GanResult(0, str(output_path), f"Video upscaled successfully to {output_path}")

    except Exception as e:
        shutil.rmtree(work_dir, ignore_errors=True)
        return GanResult(1, None, f"Video upscale error: {str(e)}")


def _run_with_spandrel_image(
    input_path: Path,
    model_name: str,
    settings: Dict[str, Any],
    base_dir: Path,
    metadata: GanModelMetadata,
    on_progress: Optional[Callable[[str], None]] = None,
    cancel_event=None
) -> GanResult:
    """Use spandrel for model loading and inference"""
    try:
        import spandrel
        import torch
        from PIL import Image
        
        if on_progress:
            on_progress("Loading model with spandrel...\n")

        model_path = base_dir / "Image_Upscale_Models" / model_name
        model = spandrel.ModelLoader().load_from_file(str(model_path))
        
        # Load image
        img = Image.open(input_path).convert('RGB')
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        img_tensor = img_tensor.to(device)
        
        # Inference
        with torch.no_grad():
            output_tensor = model(img_tensor)
        
        # Convert back to image
        output_array = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output_array = np.clip(output_array * 255.0, 0, 255).astype(np.uint8)
        output_img = Image.fromarray(output_array)
        
        # Save output
        output_format = settings.get("output_format", "png")
        output_path = collision_safe_path(input_path.with_stem(f"{input_path.stem}_upscaled").with_suffix(f".{output_format}"))
        output_img.save(output_path, quality=settings.get("output_quality", 95))
        
        return GanResult(0, str(output_path), f"Upscaled with spandrel to {output_path}")
        
    except Exception as e:
        return GanResult(1, None, f"Spandrel error: {str(e)}")


def _run_with_realesrgan_image(
    input_path: Path,
    model_name: str,
    settings: Dict[str, Any],
    base_dir: Path,
    metadata: GanModelMetadata,
    on_progress: Optional[Callable[[str], None]] = None,
    cancel_event=None
) -> GanResult:
    """Use Real-ESRGAN's inference script"""
    try:
        if on_progress:
            on_progress("Using Real-ESRGAN inference...\n")

        # Import Real-ESRGAN components
        from shared.realesrgan_runner import run_realesrgan

        # Prepare settings for Real-ESRGAN
        realesrgan_settings = {
            "input_path": str(input_path),
            "model": model_name,
            "scale": metadata.scale,
            "output_format": settings.get("output_format", "png"),
            "cuda_device": settings.get("gpu_device", "0"),
        }

        result = run_realesrgan(
            settings=realesrgan_settings,
            apply_face=False,  # We handle face restoration separately
            cancel_event=cancel_event
        )

        return GanResult(result.returncode, result.output_path, result.log)

    except Exception as e:
        return GanResult(1, None, f"Real-ESRGAN error: {str(e)}")
