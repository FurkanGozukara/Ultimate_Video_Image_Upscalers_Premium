import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, List, Tuple

from .path_utils import collision_safe_dir, collision_safe_path, normalize_path, resolve_output_location


def _has_scenedetect() -> bool:
    try:
        import scenedetect  # noqa: F401
        return True
    except Exception:
        return False


def detect_scenes(video_path: str, threshold: float = 27.0, min_scene_len: float = 2.0) -> List[Tuple[float, float]]:
    """
    Detect scenes using PySceneDetect ContentDetector.
    Returns list of (start_sec, end_sec).
    """
    if not _has_scenedetect():
        return []

    from scenedetect import open_video
    from scenedetect.detectors import ContentDetector

    video = open_video(video_path)
    scenes = video.detect_scenes(ContentDetector(threshold=threshold, min_scene_len=int(min_scene_len * video.base_fps)))
    ranges = []
    for start, end in scenes:
        ranges.append((start.get_seconds(), end.get_seconds()))
    return ranges


def fallback_scenes(video_path: str, chunk_seconds: float = 60.0, overlap_seconds: float = 0.0) -> List[Tuple[float, float]]:
    """
    Fallback to fixed-length segments using ffprobe duration with optional overlap.
    """
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        duration = float(proc.stdout.strip())
    except Exception:
        duration = chunk_seconds
    scenes = []
    start = 0.0
    step = max(1.0, chunk_seconds - overlap_seconds) if chunk_seconds > 0 else chunk_seconds
    if step <= 0:
        step = chunk_seconds
    while start < duration:
        end = min(start + chunk_seconds, duration)
        scenes.append((start, end))
        start = end - overlap_seconds if overlap_seconds > 0 else end
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


def concat_videos(chunk_paths: List[Path], output_path: Path) -> bool:
    if not chunk_paths:
        return False
    txt = output_path.parent / "concat.txt"
    with txt.open("w", encoding="utf-8") as f:
        for p in chunk_paths:
            f.write(f"file '{p.as_posix()}'\n")
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(txt), "-c", "copy", str(output_path)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode == 0 and output_path.exists()


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
) -> Tuple[int, str, str, int]:
    """
    Returns (returncode, log, final_output_path, chunk_count)
    """
    input_path = normalize_path(settings["input_path"])
    output_format = settings.get("output_format") or "mp4"
    if output_format in (None, "auto"):
        output_format = "mp4"
    work = Path(temp_dir) / "chunks"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)

    # Predict final output locations for partial/cancel handling
    predicted_final = resolve_output_location(
        input_path=input_path,
        output_format=output_format,
        global_output_dir=settings.get("output_override"),
        batch_mode=False,
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

    scenes = detect_scenes(input_path, threshold=scene_threshold, min_scene_len=min_scene_len)
    if not scenes or chunk_seconds > 0:
        effective_seconds = chunk_seconds if chunk_seconds > 0 else max(min_scene_len, 30)
        scenes = fallback_scenes(input_path, chunk_seconds=effective_seconds, overlap_seconds=max(0.0, chunk_overlap))
    on_progress(f"Detected {len(scenes)} scenes for chunking\n")

    chunk_paths = split_video(input_path, scenes, work)
    on_progress(f"Split into {len(chunk_paths)} chunks\n")

    output_chunks: List[Path] = []
    chunk_logs: List[dict] = []
    for idx, chunk in enumerate(chunk_paths, 1):
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
                        if per_chunk_cleanup:
                            shutil.rmtree(work, ignore_errors=True)
                        return 1, log_blob, str(partial_target), len(chunk_paths)
                    else:
                        partial_target = partial_video_target or collision_safe_path(work / "partial_concat.mp4")
                        ok = concat_videos(output_chunks, partial_target)
                        log_blob = f"Chunking canceled at chunk {idx}; partial output saved: {partial_target}"
                        if per_chunk_cleanup:
                            shutil.rmtree(work, ignore_errors=True)
                        if ok:
                            return 1, log_blob, str(partial_target), len(chunk_paths)
                return 1, "Canceled before processing current chunk", "", len(chunk_paths)
        except Exception:
            pass
        on_progress(f"Processing chunk {idx}/{len(chunk_paths)}: {chunk.name}\n")
        chunk_settings = settings.copy()
        chunk_settings["input_path"] = str(chunk)
        # Direct chunk outputs to temp; choose dir for PNG exports
        if output_format == "png":
            chunk_settings["output_override"] = str(work / f"{chunk.stem}_out")
        else:
            chunk_settings["output_override"] = str(work / f"{chunk.stem}_out.mp4")
        res = runner.run_seedvr2(chunk_settings, on_progress=None, preview_only=False)
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
                    if per_chunk_cleanup:
                        shutil.rmtree(work, ignore_errors=True)
                    return res.returncode, log_blob, str(partial_target), len(chunk_paths)
                else:
                    partial_target = partial_video_target or collision_safe_path(work / "partial_concat.mp4")
                    ok = concat_videos(output_chunks, partial_target)
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
            global_output_dir=settings.get("output_override"),
            batch_mode=False,
        )
        target_dir = collision_safe_dir(Path(target_dir))
        target_dir.mkdir(parents=True, exist_ok=True)
        for i, outp in enumerate(output_chunks, 1):
            dest = target_dir / f"chunk_{i:04d}"
            if Path(outp).is_dir():
                shutil.copytree(outp, dest, dirs_exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(outp, dest)
        if per_chunk_cleanup:
            shutil.rmtree(work, ignore_errors=True)
        log_blob = "Chunked processing complete (PNG)\n" + "\n".join([str(c) for c in chunk_logs])
        return 0, log_blob, str(target_dir), len(chunk_paths)

    final_path = resolve_output_location(
        input_path=input_path,
        output_format="mp4",
        global_output_dir=settings.get("output_override"),
        batch_mode=False,
    )
    final_path = collision_safe_path(Path(final_path))
    ok = concat_videos(output_chunks, final_path)
    if not ok:
        return 1, "Concat failed", str(final_path), len(chunk_paths)
    on_progress(f"Chunks concatenated to {final_path}\n")
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
    return 0, log_blob, str(final_path), len(chunk_paths)

