import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

# Supported extensions match SeedVR2 CLI
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def normalize_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    expanded = os.path.expanduser(os.path.expandvars(str(path)))
    return str(Path(expanded).resolve())


def detect_input_type(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return "missing"
    if p.is_dir():
        return "directory"
    ext = p.suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    return "unknown"


def get_media_dimensions(path: str) -> Optional[Tuple[int, int]]:
    """
    Return (width, height) for an image or video using ffprobe first, Pillow fallback.
    """
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            parts = proc.stdout.strip().split(",")
            if len(parts) >= 2:
                return int(parts[0]), int(parts[1])
    except Exception:
        pass

    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as im:
            return im.width, im.height
    except Exception:
        return None


def get_media_duration_seconds(path: str) -> Optional[float]:
    """
    Return duration in seconds for a video using ffprobe.
    """
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return float(proc.stdout.strip())
    except Exception:
        return None
    return None


def get_media_fps(path: str) -> Optional[float]:
    """Return frames per second for a video using ffprobe."""
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=r_frame_rate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            rate = proc.stdout.strip()
            if "/" in rate:
                num, den = rate.split("/")
                return float(num) / float(den)
            return float(rate)
    except Exception:
        return None
    return None

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_default_output_dir(base_dir: Path, global_settings: dict) -> Path:
    # Prefer saved global setting; fallback to ./outputs
    override = global_settings.get("output_dir")
    if override:
        return Path(normalize_path(override))
    return Path(base_dir / "outputs").resolve()


def get_default_temp_dir(base_dir: Path, global_settings: dict) -> Path:
    # Use launcher-provided TEMP/TMP if present; otherwise fallback to ./temp
    override = global_settings.get("temp_dir") or os.environ.get("TEMP") or os.environ.get("TMP")
    if override:
        return Path(normalize_path(override))
    return Path(base_dir / "temp").resolve()


def collision_safe_path(path: Path) -> Path:
    """
    Append numeric suffixes to avoid overwriting existing files or folders.
    """
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    counter = 1
    pattern = re.compile(r"(.*)_(\d{4})$")
    match = pattern.match(stem)
    if match:
        stem = match.group(1)
        counter = int(match.group(2)) + 1
    while True:
        candidate = parent / f"{stem}_{counter:04d}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def collision_safe_dir(path: Path) -> Path:
    """
    Collision-safe directory naming with _0001 suffix.
    """
    if not path.exists():
        return path
    base = path.name
    parent = path.parent
    counter = 1
    pattern = re.compile(r"(.*)_(\d{4})$")
    match = pattern.match(base)
    if match:
        base = match.group(1)
        counter = int(match.group(2)) + 1
    while True:
        candidate = parent / f"{base}_{counter:04d}"
        if not candidate.exists():
            return candidate
        counter += 1


def rife_output_path(input_path: str, png_output: bool, override: Optional[str]) -> Path:
    """
    Simple collision-safe output for RIFE:
    - If override provided, use it (file or dir) as-is.
    - Otherwise sibling <name>_rife.mp4 or .png (if png_output True).
    """
    if override:
        target = Path(normalize_path(override))
        if target.suffix == "":
            if png_output:
                ensure_dir(target)
                return collision_safe_path(target)
            target = target.with_suffix(".mp4")
        ensure_dir(target.parent)
        return collision_safe_path(target)

    inp = Path(normalize_path(input_path))
    suffix = ".png" if png_output else ".mp4"
    if png_output:
        target_dir = inp.with_name(f"{inp.stem}_rife")
        target_dir = collision_safe_dir(target_dir)
        ensure_dir(target_dir)
        return target_dir
    target = inp.with_name(f"{inp.stem}_rife{suffix}")
    ensure_dir(target.parent)
    return collision_safe_path(target)


def write_png_metadata(dir_path: Path, payload: dict):
    """
    Write metadata JSON into a PNG sequence directory.
    """
    ensure_dir(dir_path)
    meta_path = dir_path / "run_metadata.json"
    try:
        import json
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass


def generate_output_path(
    input_path: str,
    output_format: str,
    output_dir: Optional[str] = None,
    input_type: Optional[str] = None,
    from_directory: bool = False,
) -> Path:
    """
    Mirror SeedVR2 CLI generate_output_path behavior, then apply collision safety.
    """
    input_path_obj = Path(input_path)
    input_name = input_path_obj.stem

    if input_type is None:
        input_type = detect_input_type(input_path)

    # Determine base directory and suffix policy
    if output_dir:
        base_dir = Path(normalize_path(output_dir))
        add_suffix = False
    elif from_directory:
        original_dir = input_path_obj.parent
        base_dir = original_dir.parent / f"{original_dir.name}_upscaled"
        add_suffix = False
    else:
        base_dir = input_path_obj.parent
        add_suffix = True

    file_suffix = "_upscaled" if add_suffix else ""

    if output_format == "png":
        if input_type == "image":
            target = base_dir / f"{input_name}{file_suffix}.png"
            ensure_dir(target.parent)
            return collision_safe_path(target)
        else:
            target_dir = base_dir / f"{input_name}{file_suffix}"
            ensure_dir(target_dir.parent)
            target_dir = collision_safe_dir(target_dir)
            ensure_dir(target_dir)
            return target_dir
    else:
        target = base_dir / f"{input_name}{file_suffix}.mp4"
        ensure_dir(target.parent)
        return collision_safe_path(target)


def resolve_output_location(
    input_path: str,
    output_format: str,
    global_output_dir: Optional[str],
    batch_mode: bool,
) -> Path:
    """
    Resolve where outputs should go respecting CLI semantics first, then applying
    global overrides when provided.
    """
    if global_output_dir:
        return generate_output_path(
            input_path=input_path,
            output_format=output_format,
            output_dir=global_output_dir,
            input_type=detect_input_type(input_path),
            from_directory=batch_mode,
        )
    return generate_output_path(
        input_path=input_path,
        output_format=output_format,
        input_type=detect_input_type(input_path),
        from_directory=batch_mode,
    )


def emit_metadata(path: Path, payload: dict):
    """
    Write metadata JSON adjacent to output (file or directory).
    """
    target_dir = path if path.is_dir() else path.parent
    ensure_dir(target_dir)
    meta_path = target_dir / "run_metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        import json
        json.dump(payload, f, indent=2)
    return meta_path


def ffmpeg_set_fps(input_path: Path, fps: float) -> Path:
    """
    Re-mux video with a different fps (no re-encode). Falls back silently if ffmpeg missing.
    """
    if fps <= 0:
        return input_path
    if shutil.which("ffmpeg") is None:
        return input_path
    out = collision_safe_path(input_path.with_name(f"{input_path.stem}_fps{input_path.suffix}"))
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-map",
        "0",
        "-c",
        "copy",
        "-r",
        str(fps),
        str(out),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode == 0 and out.exists():
        return out
    return input_path


def is_writable(path: Path) -> bool:
    try:
        ensure_dir(path)
        test_file = path / ".write_test"
        with test_file.open("w") as f:
            f.write("ok")
        test_file.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def get_disk_free_gb(path: Path) -> float:
    try:
        usage = shutil.disk_usage(path)
        return round(usage.free / (1024 ** 3), 2)
    except Exception:
        return 0.0


