import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional, Tuple


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def has_audio_stream(path: Path) -> bool:
    """
    Return True if `path` contains at least one audio stream (ffprobe-based).
    """
    try:
        if not path.exists() or path.stat().st_size < 1024:
            return False
    except Exception:
        return False

    if not _has_ffmpeg():
        return False

    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return proc.returncode == 0 and bool((proc.stdout or "").strip())
    except Exception:
        return False


def _build_audio_args(audio_codec: str, audio_bitrate: Optional[str]) -> list[str]:
    codec = (audio_codec or "copy").strip().lower()
    if codec in ("none", "no", "off", "disable", "disabled"):
        return ["-an"]
    if codec in ("copy", "passthrough"):
        return ["-c:a", "copy"]

    args = ["-c:a", codec]
    if audio_bitrate:
        args += ["-b:a", str(audio_bitrate)]
    elif codec == "aac":
        # Sensible default for compatibility/quality.
        args += ["-b:a", "192k"]
    return args


def mux_audio(
    video_path: Path,
    audio_source_path: Path,
    output_path: Path,
    audio_codec: str = "copy",
    audio_bitrate: Optional[str] = None,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, str]:
    """
    Mux audio from `audio_source_path` into `video_path`, writing to `output_path`.
    Keeps video stream bit-exact via `-c:v copy`.
    """
    if not _has_ffmpeg():
        return False, "ffmpeg/ffprobe not available"

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    args_audio = _build_audio_args(audio_codec, audio_bitrate)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_source_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "copy",
        *args_audio,
        "-shortest",
        "-avoid_negative_ts",
        "make_zero",
    ]

    # MP4 nicety: faststart if output is mp4-ish.
    if output_path.suffix.lower() in (".mp4", ".m4v", ".mov"):
        cmd += ["-movflags", "+faststart"]

    cmd.append(str(output_path))

    if on_progress:
        on_progress(f"🔊 Muxing audio ({audio_codec}) -> {output_path.name}\n")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0 and output_path.exists():
            return True, ""
        return False, (proc.stderr or proc.stdout or "ffmpeg mux failed").strip()
    except Exception as e:
        return False, str(e)


def ensure_audio_on_video(
    video_path: Path,
    audio_source_path: Path,
    audio_codec: str = "copy",
    audio_bitrate: Optional[str] = None,
    force_replace: bool = False,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, Optional[Path], str]:
    """
    Ensure `video_path` has (or does not have) audio according to `audio_codec`.

    - If audio_codec is "none": strip audio if present.
    - Otherwise:
      - default behavior: if video already has audio, keep it;
        if it doesn't, attempt to mux from `audio_source_path`.
      - when force_replace=True: replace output audio with the source audio.
        If source has no audio, output audio is stripped to match source.

    Returns: (changed, final_path_or_None, error_message)
    """
    video_path = Path(video_path)
    audio_source_path = Path(audio_source_path)

    codec_norm = (audio_codec or "copy").strip().lower()
    want_audio = codec_norm not in ("none", "no", "off", "disable", "disabled")

    if not _has_ffmpeg():
        return False, None, "ffmpeg/ffprobe not available"

    video_has_audio = has_audio_stream(video_path)
    src_has_audio = has_audio_stream(audio_source_path)

    if not want_audio:
        if not video_has_audio:
            return False, video_path, ""
        tmp = video_path.with_name(f"{video_path.stem}.__noaudio_tmp{video_path.suffix}")
        ok, err = mux_audio(
            video_path=video_path,
            audio_source_path=audio_source_path,
            output_path=tmp,
            audio_codec="none",
            audio_bitrate=None,
            on_progress=on_progress,
        )
        if not ok:
            tmp.unlink(missing_ok=True)
            return False, None, err
        try:
            os.replace(str(tmp), str(video_path))
        except Exception:
            return True, tmp, ""
        return True, video_path, ""

    # want audio
    if force_replace:
        # Match source exactly for robust final muxing.
        if not src_has_audio:
            if not video_has_audio:
                return False, video_path, ""
            tmp = video_path.with_name(f"{video_path.stem}.__noaudio_tmp{video_path.suffix}")
            ok, err = mux_audio(
                video_path=video_path,
                audio_source_path=audio_source_path,
                output_path=tmp,
                audio_codec="none",
                audio_bitrate=None,
                on_progress=on_progress,
            )
            if not ok:
                tmp.unlink(missing_ok=True)
                return False, None, err
            try:
                os.replace(str(tmp), str(video_path))
            except Exception:
                return True, tmp, ""
            return True, video_path, ""
    else:
        if video_has_audio:
            return False, video_path, ""
        if not src_has_audio:
            return False, video_path, ""

    tmp = video_path.with_name(f"{video_path.stem}.__audio_tmp{video_path.suffix}")

    ok, err = mux_audio(
        video_path=video_path,
        audio_source_path=audio_source_path,
        output_path=tmp,
        audio_codec=audio_codec,
        audio_bitrate=audio_bitrate,
        on_progress=on_progress,
    )
    if not ok and codec_norm == "copy":
        # Robust fallback for incompatible audio codecs/containers.
        ok, err2 = mux_audio(
            video_path=video_path,
            audio_source_path=audio_source_path,
            output_path=tmp,
            audio_codec="aac",
            audio_bitrate=audio_bitrate or "192k",
            on_progress=on_progress,
        )
        if ok:
            err = ""
        else:
            err = err2 or err

    if not ok:
        tmp.unlink(missing_ok=True)
        return False, None, err

    try:
        os.replace(str(tmp), str(video_path))
    except Exception:
        # As a fallback, keep the muxed file as a sibling and return it.
        return True, tmp, ""
    return True, video_path, ""
