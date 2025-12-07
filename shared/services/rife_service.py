import queue
import subprocess
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import gradio as gr

from shared.preset_manager import PresetManager
from shared.runner import Runner
from shared.path_utils import normalize_path, ffmpeg_set_fps
from shared.face_restore import restore_video
from shared.logging_utils import RunLogger


def rife_defaults() -> Dict[str, Any]:
    return {
        "model": "rife",
        "fps_multiplier": 2.0,
        "scale": 1.0,
        "uhd_half": False,
        "png_output": False,
        "no_audio": False,
        "show_ffmpeg": False,
        "montage": False,
        "img_mode": False,
        "output_override": "",
        "model_dir": "",
        "skip_first_frames": 0,
        "load_cap": 0,
        "fps_override": 0,
        "output_format": "auto",
    }


RIFE_ORDER: List[str] = [
    "model",
    "fps_multiplier",
    "scale",
    "uhd_half",
    "png_output",
    "no_audio",
    "show_ffmpeg",
    "montage",
    "img_mode",
    "output_override",
    "model_dir",
    "skip_first_frames",
    "load_cap",
    "fps_override",
    "output_format",
]


def _rife_dict_from_args(args: List[Any]) -> Dict[str, Any]:
    return dict(zip(RIFE_ORDER, args))


def _apply_rife_preset(
    preset: Dict[str, Any],
    defaults: Dict[str, Any],
    preset_manager: PresetManager,
    current: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    base = defaults.copy()
    if current:
        base.update(current)
    merged = preset_manager.merge_config(base, preset)
    return [merged[k] for k in RIFE_ORDER]


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def build_rife_callbacks(
    preset_manager: PresetManager,
    runner: Runner,
    run_logger: RunLogger,
    global_settings: Dict[str, Any],
    output_dir: Path,
    temp_dir: Path,
):
    defaults = rife_defaults()

    def refresh_presets(model_name: str, select_name: Optional[str] = None):
        presets = preset_manager.list_presets("rife", model_name)
        last_used = preset_manager.get_last_used_name("rife", model_name)
        preferred = select_name if select_name in presets else None
        value = preferred or (last_used if last_used in presets else (presets[-1] if presets else None))
        return gr.Dropdown.update(choices=presets, value=value)

    def save_preset(preset_name: str, *args):
        if not preset_name:
            return gr.Dropdown.update(), gr.Markdown.update(value="⚠️ Enter a preset name before saving"), *list(args)
        payload = _rife_dict_from_args(list(args))
        model_name = payload["model"]
        preset_manager.save_preset("rife", model_name, preset_name, payload)
        dropdown = refresh_presets(model_name, select_name=preset_name)
        current_map = dict(zip(RIFE_ORDER, list(args)))
        loaded_vals = _apply_rife_preset(payload, defaults, preset_manager, current=current_map)
        return dropdown, gr.Markdown.update(value=f"✅ Saved preset '{preset_name}' for {model_name}"), *loaded_vals

    def load_preset(preset_name: str, model_name: str, current_values: List[Any]):
        model_name = model_name or defaults["model"]
        preset = preset_manager.load_preset("rife", model_name, preset_name)
        if preset:
            preset_manager.set_last_used("rife", model_name, preset_name)
        defaults_with_model = defaults.copy()
        defaults_with_model["model"] = model_name
        current_map = dict(zip(RIFE_ORDER, current_values))
        values = _apply_rife_preset(preset or {}, defaults_with_model, preset_manager, current=current_map)
        return values

    def safe_defaults():
        return [defaults[k] for k in RIFE_ORDER]

    def run_action(uploaded_file, img_folder, *args):
        settings_dict = _rife_dict_from_args(list(args))
        settings = {**defaults, **settings_dict}

        input_path = normalize_path(uploaded_file if uploaded_file else img_folder)
        if not input_path or not Path(input_path).exists():
            return ("❌ Input missing", "", None, "No metadata")
        settings["input_path"] = input_path
        settings["output_override"] = settings.get("output_override") or global_settings.get("output_dir", str(output_dir))

        if settings.get("output_format") == "png":
            settings["png_output"] = True
        elif settings.get("output_format") == "mp4":
            settings["png_output"] = False

        if not _ffmpeg_available():
            return ("❌ ffmpeg not found in PATH. Install ffmpeg and retry.", "", None, "No metadata")

        log_lines: List[str] = []
        progress_q: "queue.Queue[str]" = queue.Queue()

        def on_progress(line: str):
            line = line.rstrip()
            log_lines.append(line)
            progress_q.put(line)

        try:
            skip_frames = int(settings.get("skip_first_frames") or 0)
            cap_frames = int(settings.get("load_cap") or 0)
        except Exception:
            skip_frames = 0
            cap_frames = 0
        trimmed_path = None
        if (skip_frames > 0 or cap_frames > 0) and Path(settings["input_path"]).is_file():
            trimmed_path = Path(temp_dir) / f"rife_trim_{Path(settings['input_path']).stem}.mp4"
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                settings["input_path"],
            ]
            if skip_frames > 0:
                cmd.extend(["-vf", f"select='gte(n\\,{skip_frames})',setpts=PTS-STARTPTS"])
            if cap_frames > 0:
                cmd.extend(["-frames:v", str(cap_frames)])
            cmd.append(str(trimmed_path))
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if trimmed_path.exists():
                settings["input_path"] = str(trimmed_path)

        face_apply = bool(args[-1]) or global_settings.get("face_global", False)

        result_holder: Dict[str, Any] = {}

        def worker():
            try:
                result_holder["result"] = runner.run_rife(settings, on_progress=on_progress)
            except Exception as exc:
                result_holder["error"] = str(exc)
                result_holder["result"] = None

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        last_yield = time.time()
        while t.is_alive() or not progress_q.empty():
            try:
                line = progress_q.get(timeout=0.2)
                if line:
                    log_lines.append(line)
            except queue.Empty:
                pass
            now = time.time()
            if now - last_yield > 0.5:
                last_yield = now
                yield (
                    gr.Markdown.update(value="⏳ Running RIFE..."),
                    "\n".join(log_lines[-400:]),
                    None,
                    "Processing...",
                )
        t.join()

        result = result_holder.get("result")
        if result is None:
            err = result_holder.get("error", "Unknown failure")
            yield (f"❌ Failed: {err}", "\n".join(log_lines), None, "No metadata")
            return

        status = "✅ RIFE complete" if result.returncode == 0 else f"⚠️ RIFE exited with code {result.returncode}"
        out_path = result.output_path if result.output_path and result.output_path.endswith(".mp4") else None
        if settings.get("fps_override") and out_path and Path(out_path).exists():
            out_path = str(ffmpeg_set_fps(Path(out_path), float(settings["fps_override"])))
            log_lines.append(f"FPS overridden to {settings['fps_override']}")
        if face_apply and out_path and Path(out_path).exists():
            restored = restore_video(out_path, strength=0.5, on_progress=on_progress)
            if restored:
                out_path = restored
                log_lines.append(f"Face-restored video saved to {restored}")
        meta_md = f"Output: {out_path}\nReturn code: {result.returncode}"
        run_logger.write_summary(
            Path(out_path) if out_path else output_dir,
            {
                "input": input_path,
                "output": out_path,
                "returncode": result.returncode,
                "args": settings,
                "face_apply": face_apply,
                "pipeline": "rife",
            },
        )
        final_log = "\n".join(log_lines) or result.log
        yield (status, final_log, out_path, meta_md)

    return {
        "defaults": defaults,
        "order": RIFE_ORDER,
        "refresh_presets": refresh_presets,
        "save_preset": save_preset,
        "load_preset": load_preset,
        "safe_defaults": safe_defaults,
        "run_action": run_action,
    }


