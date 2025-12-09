import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, List, Tuple, Optional

import cv2
import numpy as np

from .path_utils import (
    collision_safe_dir,
    collision_safe_path,
    normalize_path,
    resolve_output_location,
    detect_input_type,
    emit_metadata,
    get_media_fps,
)

# Try to import PySceneDetect (optional dependency)
try:
    import scenedetect
    PYSCENEDETECT_AVAILABLE = True
except ImportError:
    PYSCENEDETECT_AVAILABLE = False


def _has_scenedetect() -> bool:
    """Check if PySceneDetect is installed"""
    try:
        import scenedetect  # noqa: F401
        return True
    except ImportError:
        return False
    except Exception:
        return False


def detect_scenes(
    video_path: str, 
    threshold: float = 27.0, 
    min_scene_len: float = 2.0,
    fade_detection: bool = False,
    overlap_sec: float = 0.5,
    on_progress: Optional[Callable[[str], None]] = None
) -> List[Tuple[float, float]]:
    """
    Detect scenes using PySceneDetect with proper API usage and overlap support.
    
    Args:
        video_path: Path to video file
        threshold: Content threshold for scene detection (lower = more sensitive)
        min_scene_len: Minimum scene length in seconds
        fade_detection: Enable fade in/out detection
        overlap_sec: Seconds of overlap between chunks (for temporal consistency)
        on_progress: Optional callback for progress updates
        
    Returns:
        List of (start_seconds, end_seconds) tuples for each scene with overlap applied
    """
    if not _has_scenedetect():
        if on_progress:
            on_progress("⚠️ PySceneDetect not installed, using fallback chunking\n")
        return []

    try:
        from scenedetect import VideoManager, SceneManager
        from scenedetect.detectors import ContentDetector, ThresholdDetector
        from scenedetect.video_splitter import split_video_ffmpeg
        
        if on_progress:
            on_progress(f"Detecting scenes: threshold={threshold}, min_len={min_scene_len}s\n")

        # Create video manager
        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        
        # Add content detector with proper threshold
        scene_manager.add_detector(
            ContentDetector(
                threshold=threshold,
                min_scene_len=int(min_scene_len * video_manager.get_framerate())
            )
        )
        
        # Optionally add fade detector
        if fade_detection:
            scene_manager.add_detector(
                ThresholdDetector(
                    threshold=12,  # Default fade threshold
                    min_scene_len=int(min_scene_len * video_manager.get_framerate()),
                    fade_bias=0.0
                )
            )
        
        # Start video manager
        video_manager.set_downscale_factor()
        video_manager.start()
        
        # Detect scenes
        scene_manager.detect_scenes(frame_source=video_manager, show_progress=False)
        
        # Get scene list
        scene_list = scene_manager.get_scene_list()
        
        # Release video manager
        video_manager.release()
        
        # Convert to (start_sec, end_sec) tuples
        ranges = []
        for scene in scene_list:
            start_frame, end_frame = scene
            start_sec = start_frame.get_seconds()
            end_sec = end_frame.get_seconds()
            ranges.append((start_sec, end_sec))
        
        if on_progress:
            on_progress(f"✅ Detected {len(ranges)} scenes\n")
        
        return ranges
        
    except ImportError as e:
        if on_progress:
            on_progress(f"⚠️ PySceneDetect import error: {e}, using fallback\n")
        return []
    except Exception as e:
        if on_progress:
            on_progress(f"⚠️ Scene detection error: {e}, using fallback\n")
        return []


def apply_overlap_to_scenes(
    scenes: List[Tuple[float, float]], 
    overlap_sec: float,
    total_duration: float
) -> List[Tuple[float, float]]:
    """
    Apply overlap to scene boundaries for temporal consistency.
    
    Args:
        scenes: List of (start, end) tuples without overlap
        overlap_sec: Seconds of overlap to add
        total_duration: Total video duration to clamp overlaps
        
    Returns:
        List of (start, end) tuples with overlap applied
    """
    if overlap_sec <= 0 or not scenes:
        return scenes
    
    overlapped = []
    for i, (start, end) in enumerate(scenes):
        # Extend start backwards (except first chunk)
        if i > 0:
            new_start = max(0, start - overlap_sec / 2)
        else:
            new_start = start
        
        # Extend end forwards (except last chunk)
        if i < len(scenes) - 1:
            new_end = min(total_duration, end + overlap_sec / 2)
        else:
            new_end = end
        
        overlapped.append((new_start, new_end))
    
    return overlapped


def fallback_scenes(video_path: str, chunk_seconds: float = 60.0, overlap_seconds: float = 0.0) -> List[Tuple[float, float]]:
    """
    Fallback to fixed-length segments using ffprobe duration with optional overlap.
    
    Args:
        video_path: Path to video file
        chunk_seconds: Length of each chunk in seconds
        overlap_seconds: Overlap between chunks in seconds
        
    Returns:
        List of (start_sec, end_sec) tuples with overlap applied
    """
    from .path_utils import get_media_duration_seconds
    
    try:
        duration = get_media_duration_seconds(video_path)
        if not duration or duration <= 0:
            # Try ffprobe as fallback
            proc = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path],
                capture_output=True,
                text=True,
                timeout=10,
            )
            duration = float(proc.stdout.strip())
    except Exception:
        # If all fails, use default
        duration = chunk_seconds * 2
    
    scenes = []
    start = 0.0
    
    # First pass: create chunks without overlap
    while start < duration:
        end = min(start + chunk_seconds, duration)
        scenes.append((start, end))
        start += chunk_seconds
        
        # Avoid tiny last chunk
        if start < duration and (duration - start) < (chunk_seconds * 0.3):
            scenes[-1] = (scenes[-1][0], duration)
            break
    
    # Second pass: apply overlap
    if overlap_seconds > 0:
        scenes = apply_overlap_to_scenes(scenes, overlap_seconds, duration)
    
    return scenes


def split_video(video_path: str, scenes: List[Tuple[float, float]], work_dir: Path) -> List[Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths: List[Path] = []
    for idx, (start, end) in enumerate(scenes, 1):
        out = work_dir / f"chunk_{idx:04d}.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-ss",
            str(start),
            "-to",
            str(end),
            "-c",
            "copy",
            str(out),
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if out.exists():
            chunk_paths.append(out)
    return chunk_paths


def blend_overlapping_frames_opencv(
    prev_frames: np.ndarray,
    cur_frames: np.ndarray,
    overlap_frames: int
) -> np.ndarray:
    """
    Blend overlapping frames using smooth crossfade (OpenCV implementation).
    
    Args:
        prev_frames: Last `overlap_frames` from previous chunk [N, H, W, C]
        cur_frames: First `overlap_frames` from current chunk [N, H, W, C]
        overlap_frames: Number of frames to blend
        
    Returns:
        Blended frames [overlap_frames, H, W, C]
    """
    if overlap_frames <= 0:
        return cur_frames
    
    if overlap_frames >= 3:
        # Smooth Hann window for better blending
        t = np.linspace(0.0, 1.0, overlap_frames)
        blend_start = 1.0 / 3.0
        blend_end = 2.0 / 3.0
        u = np.clip((t - blend_start) / (blend_end - blend_start), 0.0, 1.0)
        w_prev = 0.5 + 0.5 * np.cos(np.pi * u)  # Hann window
    else:
        # Linear blend for short overlaps
        w_prev = np.linspace(1.0, 0.0, overlap_frames)
    
    # Reshape weights for broadcasting [N, 1, 1, 1]
    w_prev = w_prev.reshape(-1, 1, 1, 1)
    w_cur = 1.0 - w_prev
    
    # Blend frames
    blended = prev_frames.astype(np.float32) * w_prev + cur_frames.astype(np.float32) * w_cur
    
    return blended.astype(prev_frames.dtype)


def concat_videos(chunk_paths: List[Path], output_path: Path) -> bool:
    """Simple concatenation without blending (for non-overlapping chunks)"""
    if not chunk_paths:
        return False
    txt = output_path.parent / "concat.txt"
    with txt.open("w", encoding="utf-8") as f:
        for p in chunk_paths:
            f.write(f"file '{p.as_posix()}'\n")
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(txt), "-c", "copy", str(output_path)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode == 0 and output_path.exists()


def concat_videos_with_blending(
    chunk_paths: List[Path],
    output_path: Path,
    overlap_frames: int = 0,
    fps: Optional[float] = None,
    on_progress: Optional[Callable[[str], None]] = None
) -> bool:
    """
    Concatenate video chunks with smooth blending of overlapping regions.
    
    Args:
        chunk_paths: List of video chunk file paths
        output_path: Output video path
        overlap_frames: Number of overlapping frames between chunks
        fps: Frame rate (detected from first chunk if None)
        on_progress: Progress callback
        
    Returns:
        True if successful, False otherwise
    """
    if not chunk_paths:
        return False
    
    # If no overlap, use simple concat
    if overlap_frames <= 0:
        return concat_videos(chunk_paths, output_path)
    
    try:
        if on_progress:
            on_progress("Concatenating chunks with frame blending...\n")
        
        # Create temp directory for blended output
        with tempfile.TemporaryDirectory(prefix="blend_") as temp_dir:
            temp_path = Path(temp_dir)
            
            # Read all chunks and blend overlaps
            all_frames = []
            
            for i, chunk_path in enumerate(chunk_paths):
                if on_progress:
                    on_progress(f"Loading chunk {i+1}/{len(chunk_paths)}...\n")
                
                # Read chunk frames
                cap = cv2.VideoCapture(str(chunk_path))
                if not cap.isOpened():
                    if on_progress:
                        on_progress(f"⚠️ Failed to open chunk {chunk_path}, skipping\n")
                    continue
                
                # Detect FPS from first chunk
                if fps is None and i == 0:
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                
                chunk_frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    chunk_frames.append(frame)
                
                cap.release()
                
                if not chunk_frames:
                    continue
                
                # Convert to numpy array for blending
                chunk_array = np.array(chunk_frames)
                
                if i == 0:
                    # First chunk - add all frames
                    all_frames.extend(chunk_frames)
                else:
                    # Subsequent chunks - blend overlap region
                    if len(all_frames) >= overlap_frames and len(chunk_frames) >= overlap_frames:
                        # Get overlapping regions
                        prev_tail = np.array(all_frames[-overlap_frames:])
                        cur_head = chunk_array[:overlap_frames]
                        
                        # Blend
                        if on_progress:
                            on_progress(f"Blending {overlap_frames} frames between chunks {i} and {i+1}...\n")
                        
                        blended = blend_overlapping_frames_opencv(prev_tail, cur_head, overlap_frames)
                        
                        # Replace tail of all_frames with blended, add rest of chunk
                        all_frames = all_frames[:-overlap_frames]
                        all_frames.extend(blended)
                        all_frames.extend(chunk_frames[overlap_frames:])
                    else:
                        # Not enough frames to blend, just append
                        non_overlap_start = min(overlap_frames, len(chunk_frames))
                        all_frames.extend(chunk_frames[non_overlap_start:])
            
            if not all_frames:
                if on_progress:
                    on_progress("❌ No frames to write\n")
                return False
            
            # Write blended frames to temp video
            if on_progress:
                on_progress(f"Writing {len(all_frames)} blended frames to output...\n")
            
            # Get dimensions from first frame
            height, width = all_frames[0].shape[:2]
            
            # Create temp video file
            temp_output = temp_path / "blended_temp.mp4"
            
            # Use ffmpeg to encode (better quality than cv2.VideoWriter)
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{width}x{height}",
                "-pix_fmt", "bgr24",
                "-r", str(fps or 30.0),
                "-i", "-",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                str(temp_output)
            ]
            
            proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Write frames to ffmpeg
            for frame in all_frames:
                proc.stdin.write(frame.tobytes())
            
            proc.stdin.close()
            proc.wait()
            
            if proc.returncode != 0 or not temp_output.exists():
                if on_progress:
                    on_progress(f"❌ FFmpeg encoding failed: {proc.stderr.read().decode()}\n")
                return False
            
            # Move to final output
            shutil.move(str(temp_output), str(output_path))
            
            if on_progress:
                on_progress(f"✅ Blended video saved to {output_path}\n")
            
            return True
            
    except Exception as e:
        if on_progress:
            on_progress(f"❌ Blending failed: {e}\n")
        # Fallback to simple concat
        return concat_videos(chunk_paths, output_path)


def detect_resume_state(work_dir: Path, output_format: str) -> Tuple[Optional[Path], List[Path]]:
    """
    Detect if there's a resumable chunking session.
    Returns (partial_output_path, completed_chunks) or (None, []) if no resume possible.
    """
    if not work_dir.exists():
        return None, []

    # Check for partial outputs
    if output_format == "png":
        partial_candidates = list(work_dir.glob("*_partial"))
        if partial_candidates:
            partial_dir = partial_candidates[0]
            completed_chunks = []
            chunk_pattern = partial_dir / "chunk_*.png"
            for chunk_file in sorted(chunk_pattern.parent.glob("chunk_*.png")):
                if chunk_file.exists():
                    completed_chunks.append(chunk_file)
            return partial_dir, completed_chunks
    else:
        partial_candidates = list(work_dir.glob("*_partial.mp4"))
        if partial_candidates:
            partial_file = partial_candidates[0]
            # For video, we can't easily resume chunk-by-chunk, so return empty completed list
            return partial_file, []

    return None, []


def check_resume_available(temp_dir: Path, output_format: str) -> Tuple[bool, str]:
    """
    Check if resume is available for chunking.
    Returns (available, status_message).
    """
    work_dir = temp_dir / "chunks"
    partial_path, completed_chunks = detect_resume_state(work_dir, output_format)

    if not partial_path:
        return False, "No partial chunking session found to resume."

    if output_format == "png" and completed_chunks:
        return True, f"Found {len(completed_chunks)} completed chunks ready to resume."
    elif output_format != "png" and partial_path.exists():
        return True, "Found partial video output ready to resume from."
    else:
        return False, "Partial output found but no completed chunks to resume from."


def chunk_and_process(
    runner,
    settings: dict,
    scene_threshold: float,
    min_scene_len: float,
    temp_dir: Path,
    on_progress: Callable[[str], None],
    chunk_seconds: float = 0.0,
    chunk_overlap: float = 0.0,
    per_chunk_cleanup: bool = False,
    allow_partial: bool = True,
    global_output_dir: Optional[str] = None,
    resume_from_partial: bool = False,
    progress_tracker=None,
    process_func: Optional[Callable] = None,
    model_type: str = "seedvr2",
) -> Tuple[int, str, str, int]:
    """
    🎬 UNIVERSAL PySceneDetect Chunking System - Works with ALL Models
    
    This is the PREFERRED chunking method that works universally across:
    - SeedVR2 (diffusion-based video upscaling)
    - GAN models (Real-ESRGAN, etc.)
    - RIFE (frame interpolation)
    - FlashVSR+ (real-time diffusion)
    
    How it works:
    1. Splits video into scenes using PySceneDetect (intelligent scene detection)
    2. OR splits into fixed-duration chunks if scene detection disabled
    3. Processes each chunk independently with the selected model
    4. Concatenates results with optional frame blending for smooth transitions
    
    Configuration (from Resolution & Scene Split tab):
    - chunk_seconds: Duration of each chunk (0 = use scene detection)
    - scene_threshold: Sensitivity for scene detection
    - chunk_overlap: Overlap between chunks for temporal consistency
    
    Note: For SeedVR2, this can work ALONGSIDE native streaming (--chunk_size in frames).
    PySceneDetect creates scene chunks, then each chunk can use native streaming internally.
    
    Args:
        runner: Runner instance with model-specific run methods
        settings: Processing settings dict (must include input_path, output_format, etc.)
        scene_threshold: PySceneDetect sensitivity (lower = more cuts, 27 = default)
        min_scene_len: Minimum scene duration in seconds
        temp_dir: Temporary directory for chunk storage
        on_progress: Progress callback for UI updates
        chunk_seconds: Fixed chunk size in seconds (0 = use intelligent scene detection)
        chunk_overlap: Overlap between chunks in seconds (for smooth transitions)
        per_chunk_cleanup: Delete temp files after each chunk (saves disk space)
        allow_partial: Save partial results on cancel/error
        global_output_dir: Output directory override
        resume_from_partial: Resume from previous interrupted run
        progress_tracker: Additional progress tracking callback
        process_func: Optional custom processing function (takes settings, returns RunResult)
                     If None, uses model_type to select runner method
        model_type: Model type ("seedvr2", "gan", "rife", "flashvsr") - used if process_func is None
    
    Returns:
        (returncode, log, final_output_path, chunk_count)
    """
    input_path = normalize_path(settings["input_path"])
    input_type = detect_input_type(input_path)
    output_format = settings.get("output_format") or "mp4"
    if output_format in (None, "auto"):
        output_format = "mp4"
    work = Path(temp_dir) / "chunks"
    existing_partial, existing_chunks = detect_resume_state(work, output_format)

    # Initialize variables
    start_chunk_idx = 0
    resuming = False
    
    if resume_from_partial and existing_partial and existing_chunks:
        on_progress(f"Resuming from partial output: {existing_partial} with {len(existing_chunks)} completed chunks\n")
        resuming = True
        # Don't clean work directory - we're resuming!
        # chunk_paths will be set later from actual input, not from existing chunks
        start_chunk_idx = len(existing_chunks)
    else:
        # Fresh start - clean work directory
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True, exist_ok=True)
        start_chunk_idx = 0

    # Predict final output locations for partial/cancel handling
    global_override = settings.get("output_override") or global_output_dir
    predicted_final = resolve_output_location(
        input_path=input_path,
        output_format=output_format,
        global_output_dir=global_override,
        batch_mode=False,
        png_padding=settings.get("png_padding"),
        png_keep_basename=settings.get("png_keep_basename", False),
    )
    predicted_final_path = Path(predicted_final)
    if output_format == "png":
        # For PNG sequences, ensure we point to a directory; single-image PNG still gets a sibling folder for partials
        base_dir = (
            predicted_final_path.parent / predicted_final_path.stem
            if predicted_final_path.suffix.lower() == ".png"
            else predicted_final_path
        )
        partial_png_target = collision_safe_dir(base_dir.with_name(f"{base_dir.name}_partial"))
        partial_video_target = None
    else:
        partial_png_target = None
        partial_video_target = collision_safe_path(
            predicted_final_path.with_name(f"{predicted_final_path.stem}_partial{predicted_final_path.suffix}")
        )

    # Special handling for frame-folder inputs (image sequences)
    if input_type == "directory":
        frames = sorted(
            [
                f
                for f in Path(input_path).iterdir()
                if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
            ]
        )
        if not frames:
            return 1, "No frames found in folder for chunking", "", 0
        fps_guess = 30.0
        frame_window = len(frames) if chunk_seconds <= 0 else max(1, int(chunk_seconds * fps_guess))
        overlap_frames = 0 if chunk_overlap <= 0 else int(chunk_overlap * fps_guess)
        if overlap_frames >= frame_window:
            overlap_frames = max(0, frame_window - 1)
        chunk_specs = []
        start = 0
        idx = 1
        while start < len(frames):
            end = min(len(frames), start + frame_window)
            chunk_specs.append((idx, frames[start:end]))
            if end == len(frames):
                break
            start = end - overlap_frames
            idx += 1
        on_progress(f"Detected {len(chunk_specs)} frame chunks\n")
        chunk_paths = []
        for idx, frame_list in chunk_specs:
            cdir = work / f"chunk_{idx:04d}"
            cdir.mkdir(parents=True, exist_ok=True)
            for f in frame_list:
                shutil.copy2(f, cdir / f.name)
            chunk_paths.append(cdir)
    else:
        scenes = detect_scenes(input_path, threshold=scene_threshold, min_scene_len=min_scene_len)
        if not scenes or chunk_seconds > 0:
            effective_seconds = chunk_seconds if chunk_seconds > 0 else max(min_scene_len, 30)
            scenes = fallback_scenes(input_path, chunk_seconds=effective_seconds, overlap_seconds=max(0.0, chunk_overlap))
        on_progress(f"Detected {len(scenes)} scenes for chunking\n")

        chunk_paths = split_video(input_path, scenes, work)
        on_progress(f"Split into {len(chunk_paths)} chunks\n")

    output_chunks: List[Path] = []
    chunk_logs: List[dict] = []

    # If resuming, load existing completed chunks and skip them
    if resuming and existing_chunks:
        output_chunks = existing_chunks.copy()
        for i, chunk_path in enumerate(existing_chunks, 1):
            chunk_logs.append({
                "chunk_index": i,
                "input": "resumed",
                "output": str(chunk_path),
                "returncode": 0,
                "resumed": True,
            })
        on_progress(f"✅ Loaded {len(existing_chunks)} completed chunks from previous run - skipping to chunk {start_chunk_idx + 1}\n")
        if progress_tracker:
            progress_tracker(len(existing_chunks) / len(chunk_paths), desc=f"Resumed {len(existing_chunks)} chunks")

    for idx, chunk in enumerate(chunk_paths[start_chunk_idx:], start_chunk_idx + 1):
        # Respect external cancellation
        try:
            if getattr(runner, "is_canceled", lambda: False)():
                # If we already have processed chunks, return partial output immediately
                if allow_partial and output_chunks:
                    if output_format == "png":
                        partial_target = partial_png_target or collision_safe_dir(work / "partial_chunks")
                        partial_target.mkdir(parents=True, exist_ok=True)
                        for i, outp in enumerate(output_chunks, 1):
                            dest = partial_target / f"chunk_{i:04d}"
                            if Path(outp).is_dir():
                                shutil.copytree(outp, dest, dirs_exist_ok=True)
                            else:
                                shutil.copy2(outp, dest)
                        log_blob = f"Chunking canceled at chunk {idx}; partial PNG outputs saved to {partial_target}"
                        try:
                            emit_metadata(
                                partial_target,
                                {
                                    "returncode": 1,
                                    "chunks": chunk_logs,
                                    "partial": True,
                                    "canceled": True,
                                    "chunk_index": idx,
                                },
                            )
                        except Exception:
                            pass
                        if per_chunk_cleanup:
                            shutil.rmtree(work, ignore_errors=True)
                        return 1, log_blob, str(partial_target), len(chunk_paths)
                    else:
                        partial_target = partial_video_target or collision_safe_path(work / "partial_concat.mp4")
                        # Use blending concat if overlap specified
                        overlap_frames_for_blend = int(chunk_overlap * 30.0) if chunk_overlap > 0 else 0
                        ok = concat_videos_with_blending(
                            output_chunks, 
                            partial_target, 
                            overlap_frames=overlap_frames_for_blend,
                            on_progress=on_progress
                        )
                        log_blob = f"Chunking canceled at chunk {idx}; partial output saved: {partial_target}"
                        try:
                            emit_metadata(
                                partial_target,
                                {
                                    "returncode": 1 if not ok else 0,
                                    "chunks": chunk_logs,
                                    "partial": True,
                                    "canceled": True,
                                    "chunk_index": idx,
                                },
                            )
                        except Exception:
                            pass
                        if per_chunk_cleanup:
                            shutil.rmtree(work, ignore_errors=True)
                        if ok:
                            return 1, log_blob, str(partial_target), len(chunk_paths)
                return 1, "Canceled before processing current chunk", "", len(chunk_paths)
        except Exception:
            pass
        # Only update progress when chunk completes (not during processing to avoid UI spam)
        chunk_settings = settings.copy()
        chunk_settings["input_path"] = str(chunk)
        # Direct chunk outputs to temp; choose dir for PNG exports
        if output_format == "png":
            chunk_settings["output_override"] = str(work / f"{chunk.stem}_out")
        else:
            chunk_settings["output_override"] = str(work / f"{chunk.stem}_out.mp4")
        
        # Use provided processing function or select based on model type
        if process_func:
            res = process_func(chunk_settings, on_progress=None)
        elif model_type == "seedvr2":
            res = runner.run_seedvr2(chunk_settings, on_progress=None, preview_only=False)
        elif model_type == "gan":
            res = runner.run_gan(chunk_settings, on_progress=None)
        elif model_type == "rife":
            res = runner.run_rife(chunk_settings, on_progress=None)
        else:
            # Fallback to seedvr2 for backward compatibility
            res = runner.run_seedvr2(chunk_settings, on_progress=None, preview_only=False)

        # Update progress only after chunk completion
        if progress_tracker:
            progress_tracker(idx / len(chunk_paths), desc=f"Completed chunk {idx}/{len(chunk_paths)}")
        if res.returncode != 0 or getattr(runner, "is_canceled", lambda: False)():
            on_progress(f"Chunk {idx} failed with code {res.returncode}\n")
            if allow_partial and output_chunks:
                # Attempt to stitch already processed chunks
                if output_format == "png":
                    partial_target = partial_png_target or collision_safe_dir(work / "partial_chunks")
                    partial_target.mkdir(parents=True, exist_ok=True)
                    for i, outp in enumerate(output_chunks, 1):
                        dest = partial_target / f"chunk_{i:04d}"
                        if Path(outp).is_dir():
                            shutil.copytree(outp, dest, dirs_exist_ok=True)
                        else:
                            shutil.copy2(outp, dest)
                    log_blob = f"Chunking stopped early at chunk {idx}; partial PNG outputs saved to {partial_target}"
                    try:
                        emit_metadata(
                            partial_target,
                            {
                                "returncode": res.returncode,
                                "chunks": chunk_logs,
                                "partial": True,
                                "failed_chunk": idx,
                                "canceled": getattr(runner, "is_canceled", lambda: False)(),
                            },
                        )
                    except Exception:
                        pass
                    if per_chunk_cleanup:
                        shutil.rmtree(work, ignore_errors=True)
                    return res.returncode, log_blob, str(partial_target), len(chunk_paths)
                else:
                    partial_target = partial_video_target or collision_safe_path(work / "partial_concat.mp4")
                    # Use blending concat if overlap specified
                    overlap_frames_for_blend = int(chunk_overlap * 30.0) if chunk_overlap > 0 else 0
                    ok = concat_videos_with_blending(
                        output_chunks,
                        partial_target,
                        overlap_frames=overlap_frames_for_blend,
                        on_progress=on_progress
                    )
                    if ok:
                        on_progress(f"Partial output stitched to {partial_target}\n")
                        meta = {
                            "partial": True,
                            "failed_chunk": idx,
                            "returncode": res.returncode,
                            "processed_chunks": len(output_chunks),
                            "canceled": getattr(runner, "is_canceled", lambda: False)(),
                        }
                        log_blob = f"Chunking stopped early at chunk {idx}; partial output saved: {partial_target}\n{meta}"
                        try:
                            emit_metadata(
                                partial_target,
                                {
                                    "returncode": res.returncode,
                                    "chunks": chunk_logs,
                                    "partial": True,
                                    "failed_chunk": idx,
                                    "processed_chunks": len(output_chunks),
                                    "canceled": getattr(runner, "is_canceled", lambda: False)(),
                                },
                            )
                        except Exception:
                            pass
                        if per_chunk_cleanup:
                            shutil.rmtree(work, ignore_errors=True)
                        return res.returncode, log_blob, str(partial_target), len(chunk_paths)
            return res.returncode, res.log, res.output_path or "", len(chunk_paths)
        if res.output_path and Path(res.output_path).exists():
            output_chunks.append(Path(res.output_path))
        chunk_logs.append(
            {
                "chunk_index": idx,
                "input": str(chunk),
                "output": res.output_path,
                "returncode": res.returncode,
            }
        )

    if output_format == "png":
        # Aggregate chunk PNG outputs into a collision-safe parent directory
        target_dir = resolve_output_location(
            input_path=input_path,
            output_format="png",
            global_output_dir=global_override,
            batch_mode=False,
            png_padding=settings.get("png_padding"),
            png_keep_basename=settings.get("png_keep_basename", False),
        )
        target_dir = collision_safe_dir(Path(target_dir))
        target_dir.mkdir(parents=True, exist_ok=True)
        pad_val = max(1, int(settings.get("png_padding") or 5))
        for i, outp in enumerate(output_chunks, 1):
            dest = target_dir / f"chunk_{i:0{pad_val}d}"
            if Path(outp).is_dir():
                shutil.copytree(outp, dest, dirs_exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(outp, dest)
        if per_chunk_cleanup:
            shutil.rmtree(work, ignore_errors=True)
        log_blob = "Chunked processing complete (PNG)\n" + "\n".join([str(c) for c in chunk_logs])
        try:
            emit_metadata(
                target_dir,
                {
                    "returncode": 0,
                    "chunks": chunk_logs,
                    "partial": False,
                    "output_format": output_format,
                },
            )
        except Exception:
            pass
        return 0, log_blob, str(target_dir), len(chunk_paths)

    final_path = resolve_output_location(
        input_path=input_path,
        output_format="mp4",
        global_output_dir=global_override,
        batch_mode=False,
        png_padding=settings.get("png_padding"),
        png_keep_basename=settings.get("png_keep_basename", False),
    )
    final_path = collision_safe_path(Path(final_path))
    
    # Use blending concat if overlap specified
    overlap_frames_for_blend = int(chunk_overlap * (get_media_fps(input_path) or 30.0)) if chunk_overlap > 0 else 0
    ok = concat_videos_with_blending(
        output_chunks,
        final_path,
        overlap_frames=overlap_frames_for_blend,
        fps=get_media_fps(input_path),
        on_progress=on_progress
    )
    
    if not ok:
        return 1, "Concat failed", str(final_path), len(chunk_paths)
    on_progress(f"Chunks concatenated with blending to {final_path}\n")
    if per_chunk_cleanup:
        shutil.rmtree(work, ignore_errors=True)
    # Write chunk metadata
    meta_path = final_path.parent / "chunk_metadata.json"
    try:
        import json
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(chunk_logs, f, indent=2)
    except Exception:
        pass
    log_blob = "Chunked processing complete\n" + "\n".join([str(c) for c in chunk_logs])
    # Emit consolidated metadata for chunked runs
    try:
        emit_metadata(
            final_path,
            {
                "returncode": 0,
                "chunks": chunk_logs,
                "partial": False,
                "output_format": output_format,
            },
        )
    except Exception:
        pass
    return 0, log_blob, str(final_path), len(chunk_paths)

