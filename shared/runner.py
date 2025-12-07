import io
import os
import platform
import runpy
import signal
import subprocess
import sys
import threading
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .path_utils import (
    emit_metadata,
    ffmpeg_set_fps,
    normalize_path,
    resolve_output_location,
    rife_output_path,
    write_png_metadata,
    detect_input_type,
)
from .model_manager import get_model_manager, ModelType

try:
    import torch  # type: ignore
except Exception:
    torch = None  # runtime check fallback


class RunResult:
    def __init__(self, returncode: int, output_path: Optional[str], log: str):
        self.returncode = returncode
        self.output_path = output_path
        self.log = log


class Runner:
    """
    Wrapper for invoking model CLIs with cancellation support.

    Defaults to subprocess mode. An "in-app" mode executes inline (no subprocess, not cancelable mid-run).
    """

    def __init__(self, base_dir: Path, temp_dir: Path, output_dir: Path, telemetry_enabled: bool = True):
        self.base_dir = Path(base_dir)
        self.temp_dir = Path(temp_dir)
        self.output_dir = Path(output_dir)
        self._lock = threading.Lock()
        self._active_process: Optional[subprocess.Popen] = None
        self._active_mode = "subprocess"
        self._log_lines: List[str] = []
        self._canceled = False
        self._telemetry_enabled = telemetry_enabled
        self._last_model_id: Optional[str] = None
        self._model_manager = get_model_manager()

    # ------------------------------------------------------------------ #
    # Mode management
    # ------------------------------------------------------------------ #
    def set_mode(self, mode: str):
        if mode not in ("subprocess", "in_app"):
            raise ValueError("Invalid mode")
        self._active_mode = mode

    def get_mode(self) -> str:
        return self._active_mode

    def set_telemetry(self, enabled: bool):
        """Toggle metadata emission (run summaries) at runtime."""
        self._telemetry_enabled = bool(enabled)

    # ------------------------------------------------------------------ #
    # Cancellation
    # ------------------------------------------------------------------ #
    def cancel(self) -> bool:
        with self._lock:
            proc = self._active_process
            self._canceled = True
        if not proc:
            return False
        try:
            if platform.system() == "Windows":
                # Send CTRL_BREAK to the process group if possible
                proc.send_signal(signal.CTRL_BREAK_EVENT)
                time.sleep(0.5)
                proc.terminate()
            else:
                proc.terminate()
                time.sleep(0.5)
                if proc.poll() is None:
                    proc.kill()
            return True
        finally:
            with self._lock:
                self._active_process = None

    def is_canceled(self) -> bool:
        with self._lock:
            return self._canceled

    # ------------------------------------------------------------------ #
    # SeedVR2 runner
    # ------------------------------------------------------------------ #
    def run_seedvr2(
        self,
        settings: Dict[str, Any],
        on_progress: Optional[Callable[[str], None]] = None,
        preview_only: bool = False,
    ) -> RunResult:
        """
        Run SeedVR2 CLI with given settings.

        settings should include all CLI-relevant keys aligned with inference_cli.py.
        """
        cli_path = self.base_dir / "SeedVR2" / "inference_cli.py"
        if not cli_path.exists():
            raise FileNotFoundError(f"SeedVR2 CLI not found at {cli_path}")

        input_path = normalize_path(settings.get("input_path"))
        if not input_path:
            raise ValueError("Input path is required.")

        output_format = settings.get("output_format") or "auto"
        format_for_cli = None if output_format == "auto" else output_format

        batch_mode = bool(settings.get("batch_mode"))
        # Only honor an explicit override; otherwise let resolve_output_location
        # derive the path using global output dir with _upscaled suffix semantics.
        effective_output_override = settings.get("output_override") or None
        if preview_only:
            # Force single-frame preview: load_cap=1, image path remains same
            settings = settings.copy()
            settings["load_cap"] = 1

        # Respect CLI auto-detect: if format is None/auto, choose based on input type
        if preview_only:
            effective_output_format = "png"
        elif format_for_cli is None:
            itype = detect_input_type(input_path)
            effective_output_format = "png" if itype in ("image", "directory") else "mp4"
        else:
            effective_output_format = format_for_cli

        # Predict output path: if user supplied an explicit file path, keep it; otherwise mirror CLI logic
        predicted_output: Optional[Path]
        global_override: Optional[str] = None
        if effective_output_override:
            override_path = Path(normalize_path(effective_output_override))
            if override_path.suffix:
                predicted_output = override_path
            else:
                global_override = str(override_path)
                predicted_output = resolve_output_location(
                    input_path=input_path,
                    output_format=effective_output_format,
                    global_output_dir=global_override,
                    batch_mode=batch_mode,
                    png_padding=settings.get("png_padding"),
                    png_keep_basename=settings.get("png_keep_basename", False),
                )
        else:
            global_override = str(self.output_dir)
            predicted_output = resolve_output_location(
                input_path=input_path,
                output_format=effective_output_format,
                global_output_dir=global_override,
                batch_mode=batch_mode,
                png_padding=settings.get("png_padding"),
                png_keep_basename=settings.get("png_keep_basename", False),
            )

        cmd = self._build_seedvr2_cmd(cli_path, settings, format_for_cli, preview_only, output_override=effective_output_override)

        # Compile guardrail: if compile requested on Windows but vcvars is missing, disable compile flags with a warning.
        if platform.system() == "Windows" and (settings.get("compile_dit") or settings.get("compile_vae")):
            from .health import is_vs_build_tools_available
            if not is_vs_build_tools_available():
                warn_msg = "⚠️ VS Build Tools not found; torch.compile requires Visual Studio Build Tools. Install 'Desktop development with C++' workload."
                if on_progress:
                    on_progress(f"{warn_msg}\n")
                settings["compile_dit"] = False
                settings["compile_vae"] = False
                cmd = [c for c in cmd if c not in ("--compile_dit", "--compile_vae")]

        # In-app mode: execute within current process to avoid subprocess overhead (not cancelable)
        if self._active_mode == "in_app":
            self._maybe_clear_in_app_model(settings.get("dit_model"))
            return self._run_seedvr2_in_app(cli_path, cmd, predicted_output, settings, on_progress)

        cmd = self._maybe_wrap_with_vcvars(cmd, settings)

        # For subprocess mode, preload model to ensure it's ready (warm up the model cache)
        if self._active_mode == "subprocess":
            if on_progress:
                on_progress("Preloading model for subprocess execution...\n")
            model_ready = self.ensure_seedvr2_model_loaded(settings, on_progress)
            if not model_ready:
                return RunResult(1, None, "Failed to preload required model")
            if on_progress:
                on_progress("Model ready, starting subprocess...\n")

        env = os.environ.copy()
        env["TEMP"] = str(self.temp_dir)
        env["TMP"] = str(self.temp_dir)
        env.setdefault("PYTHONWARNINGS", "ignore")

        # Ensure base dirs exist
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        creationflags = 0
        preexec_fn = None
        if platform.system() == "Windows":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            preexec_fn = os.setsid  # type: ignore[arg-type]

        proc: Optional[subprocess.Popen] = None
        log_lines: List[str] = []
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                cwd=self.base_dir,
                creationflags=creationflags,
                preexec_fn=preexec_fn,
            )
            with self._lock:
                self._active_process = proc
                self._canceled = False

            assert proc.stdout is not None
            for line in proc.stdout:
                log_lines.append(line.rstrip())
                if on_progress:
                    on_progress(line)
                # Stop early if canceled
                with self._lock:
                    if self._active_process is None:
                        break
            proc.wait()
        finally:
            with self._lock:
                self._active_process = None

        output_path = str(predicted_output) if predicted_output else None
        if self._canceled and predicted_output and Path(predicted_output).exists():
            log_lines.append("Run canceled; partial output preserved.")
            output_path = str(predicted_output)
        # Emit simple metadata
        if output_path and self._telemetry_enabled:
            emit_metadata(
                Path(output_path),
                {
                    "returncode": proc.returncode if proc else -1,
                    "output": output_path,
                    "args": settings,
                },
            )
        self._last_model_id = settings.get("dit_model", self._last_model_id)
        return RunResult(proc.returncode if proc else -1, output_path, "\n".join(log_lines))

    def _run_seedvr2_in_app(self, cli_path: Path, cmd: List[str], predicted_output: Optional[Path], settings: Dict[str, Any], on_progress: Optional[Callable[[str], None]] = None) -> RunResult:
        """
        Execute SeedVR2 CLI inline (single process). Not cancelable mid-run.
        Uses pre-loaded model from model manager.
        """
        # Ensure model is loaded
        dit_model = settings.get("dit_model", "")
        if not self.ensure_seedvr2_model_loaded(settings, on_progress):
            error_msg = f"Failed to load SeedVR2 model: {dit_model}"
            return RunResult(1, None, error_msg)

        # Get the pre-loaded runner
        runner_result = self._model_manager.get_model_runner(
            ModelType.SEEDVR2, dit_model, **settings
        )

        if not runner_result:
            error_msg = f"Model not available: {dit_model}"
            return RunResult(1, None, error_msg)

        runner, cache_context = runner_result

        # For now, fall back to CLI execution since the runner integration is complex
        # TODO: Implement direct runner execution for better performance
        script_args = cmd[1:]  # drop python executable
        buf = io.StringIO()
        rc = 0
        try:
            with redirect_stdout(buf), redirect_stderr(buf):
                sys.argv = script_args
                runpy.run_path(str(cli_path), run_name="__main__")
        except SystemExit as exc:  # argparse exits
            rc = int(exc.code) if isinstance(exc.code, int) else 1
        except Exception as exc:
            buf.write(f"{exc}\n")
            rc = 1
        finally:
            sys.argv = [sys.executable]

        output_path = str(predicted_output) if predicted_output else None
        if output_path and self._telemetry_enabled:
            emit_metadata(
                Path(output_path),
                {
                    "returncode": rc,
                    "output": output_path,
                    "args": settings,
                },
            )
        self._last_model_id = dit_model
        return RunResult(rc, output_path, buf.getvalue())

    def ensure_seedvr2_model_loaded(self, settings: Dict[str, Any], on_progress: Optional[Callable[[str], None]] = None) -> bool:
        """
        Ensure the required SeedVR2 model is loaded, loading it if necessary.

        Returns True if model is ready, False if loading failed.
        """
        dit_model = settings.get("dit_model", "")
        if not dit_model:
            return False

        # Check if model is already loaded
        if self._model_manager.is_model_loaded(ModelType.SEEDVR2, dit_model, **settings):
            return True

        # Model not loaded, need to load it
        def load_callback():
            # Import here to avoid circular imports
            from ..SeedVR2.src.core.generation_utils import setup_generation_context, prepare_runner
            from ..SeedVR2.src.utils.debug import Debug

            debug = Debug(enabled=False)  # Minimal debug for loading

            # Setup generation context (similar to CLI)
            device_list = [settings.get("cuda_device", "0")]
            platform_type = "cuda" if torch and torch.cuda.is_available() else "cpu"

            # Parse devices
            if isinstance(device_list, str):
                device_list = [d.strip() for d in device_list.split(",") if d.strip()]

            inference_device = self._device_id_to_name(device_list[0], platform_type)

            # Setup context
            ctx = setup_generation_context(
                dit_device=inference_device,
                vae_device=inference_device,
                dit_offload_device=settings.get("dit_offload_device", "none"),
                vae_offload_device=settings.get("vae_offload_device", "none"),
                tensor_offload_device=settings.get("tensor_offload_device", "cpu"),
                debug=debug
            )

            # Build torch compile args
            torch_compile_args_dit = None
            if settings.get("compile_dit"):
                torch_compile_args_dit = {
                    "backend": settings.get("compile_backend", "inductor"),
                    "mode": settings.get("compile_mode", "default"),
                    "fullgraph": settings.get("compile_fullgraph", False),
                    "dynamic": settings.get("compile_dynamic", False),
                    "dynamo_cache_size_limit": settings.get("compile_dynamo_cache_size_limit", 64),
                    "dynamo_recompile_limit": settings.get("compile_dynamo_recompile_limit", 128),
                }

            # Prepare runner
            model_dir = settings.get("model_dir") or str(self.base_dir / "models" / "SeedVR2")

            runner, cache_context = prepare_runner(
                dit_model=dit_model,
                vae_model="ema_vae_fp16.safetensors",  # Default VAE
                model_dir=model_dir,
                debug=debug,
                ctx=ctx,
                dit_cache=False,  # Disable caching for now, let model manager handle it
                vae_cache=False,
                block_swap_config={
                    'blocks_to_swap': settings.get('blocks_to_swap', 0),
                    'swap_io_components': settings.get('swap_io_components', False),
                    'offload_device': settings.get('dit_offload_device', 'none'),
                },
                encode_tiled=settings.get('vae_encode_tiled', False),
                encode_tile_size=(settings.get('vae_encode_tile_size', 1024), settings.get('vae_encode_tile_size', 1024)),
                encode_tile_overlap=(settings.get('vae_encode_tile_overlap', 128), settings.get('vae_encode_tile_overlap', 128)),
                decode_tiled=settings.get('vae_decode_tiled', False),
                decode_tile_size=(settings.get('vae_decode_tile_size', 1024), settings.get('vae_decode_tile_size', 1024)),
                decode_tile_overlap=(settings.get('vae_decode_tile_overlap', 128), settings.get('vae_decode_tile_overlap', 128)),
                tile_debug=settings.get('tile_debug', 'false'),
                attention_mode=settings.get('attention_mode', 'sdpa'),
                torch_compile_args_dit=torch_compile_args_dit,
            )

            return runner, cache_context

        # Load the model using the model manager
        success = self._model_manager.preload_model(
            ModelType.SEEDVR2,
            dit_model,
            load_callback,
            on_progress,
            **settings
        )

        return success

    # ------------------------------------------------------------------ #
    # Command builder
    # ------------------------------------------------------------------ #
    def _find_vcvars(self) -> Optional[Path]:
        """
        Try to locate vcvarsall.bat in common VS Build Tools locations.
        """
        candidates = [
            Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"),
            Path(r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"),
            Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"),
            Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"),
        ]
        # Check environment variable first
        vs_install_dir = os.environ.get("VSINSTALLDIR")
        if vs_install_dir:
            env_candidate = Path(vs_install_dir) / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
            if env_candidate.exists():
                return env_candidate

        # Check candidates
        for candidate in candidates:
            if candidate.exists():
                # Quick validation that it's actually a vcvarsall.bat file
                try:
                    with open(candidate, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read(200)
                        if 'vcvarsall.bat' in content or '@echo off' in content:
                            return candidate
                except Exception:
                    continue

        return None

    def _build_seedvr2_cmd(
        self,
        cli_path: Path,
        settings: Dict[str, Any],
        output_format: Optional[str],
        preview_only: bool,
        output_override: Optional[str],
    ) -> List[str]:
        cmd: List[str] = [sys.executable, str(cli_path), settings["input_path"]]

        # Output path override
        if output_override:
            cmd.extend(["--output", output_override])

        if output_format:
            cmd.extend(["--output_format", output_format])

        if settings.get("model_dir"):
            cmd.extend(["--model_dir", settings["model_dir"]])

        if settings.get("dit_model"):
            cmd.extend(["--dit_model", settings["dit_model"]])

        # Processing params
        def _add_int(flag: str, key: str):
            if settings.get(key) is not None:
                cmd.extend([flag, str(int(settings[key]))])

        _add_int("--resolution", "resolution")
        _add_int("--max_resolution", "max_resolution")
        _add_int("--batch_size", "batch_size")
        if settings.get("uniform_batch_size"):
            cmd.append("--uniform_batch_size")
        _add_int("--seed", "seed")
        _add_int("--skip_first_frames", "skip_first_frames")
        _add_int("--load_cap", "load_cap")
        _add_int("--prepend_frames", "prepend_frames")
        _add_int("--temporal_overlap", "temporal_overlap")

        # Quality
        if settings.get("color_correction"):
            cmd.extend(["--color_correction", settings["color_correction"]])
        if settings.get("input_noise_scale") is not None:
            cmd.extend(["--input_noise_scale", str(settings["input_noise_scale"])])
        if settings.get("latent_noise_scale") is not None:
            cmd.extend(["--latent_noise_scale", str(settings["latent_noise_scale"])])

        # Devices
        if settings.get("cuda_device"):
            cmd.extend(["--cuda_device", settings["cuda_device"]])
        if settings.get("dit_offload_device"):
            cmd.extend(["--dit_offload_device", settings["dit_offload_device"]])
        if settings.get("vae_offload_device"):
            cmd.extend(["--vae_offload_device", settings["vae_offload_device"]])
        if settings.get("tensor_offload_device"):
            cmd.extend(["--tensor_offload_device", settings["tensor_offload_device"]])

        # BlockSwap
        _add_int("--blocks_to_swap", "blocks_to_swap")
        if settings.get("swap_io_components"):
            cmd.append("--swap_io_components")

        # VAE tiling
        if settings.get("vae_encode_tiled"):
            cmd.append("--vae_encode_tiled")
        _add_int("--vae_encode_tile_size", "vae_encode_tile_size")
        _add_int("--vae_encode_tile_overlap", "vae_encode_tile_overlap")
        if settings.get("vae_decode_tiled"):
            cmd.append("--vae_decode_tiled")
        _add_int("--vae_decode_tile_size", "vae_decode_tile_size")
        _add_int("--vae_decode_tile_overlap", "vae_decode_tile_overlap")
        if settings.get("tile_debug") and settings["tile_debug"] != "false":
            cmd.extend(["--tile_debug", settings["tile_debug"]])

        # Performance
        if settings.get("attention_mode"):
            cmd.extend(["--attention_mode", settings["attention_mode"]])
        if settings.get("compile_dit"):
            cmd.append("--compile_dit")
        if settings.get("compile_vae"):
            cmd.append("--compile_vae")
        if settings.get("compile_backend"):
            cmd.extend(["--compile_backend", settings["compile_backend"]])
        if settings.get("compile_mode"):
            cmd.extend(["--compile_mode", settings["compile_mode"]])
        if settings.get("compile_fullgraph"):
            cmd.append("--compile_fullgraph")
        if settings.get("compile_dynamic"):
            cmd.append("--compile_dynamic")
        _add_int("--compile_dynamo_cache_size_limit", "compile_dynamo_cache_size_limit")
        _add_int("--compile_dynamo_recompile_limit", "compile_dynamo_recompile_limit")

        # Caching
        if settings.get("cache_dit"):
            cmd.append("--cache_dit")
        if settings.get("cache_vae"):
            cmd.append("--cache_vae")

        # Debug
        if settings.get("debug"):
            cmd.append("--debug")

        # Preview: prefer PNG for quick visualization
        if preview_only and not output_format:
            cmd.extend(["--output_format", "png"])

        return cmd

    def _maybe_wrap_with_vcvars(self, cmd: List[str], settings: Dict[str, Any]) -> List[str]:
        """
        On Windows, if compile is requested, attempt to call vcvarsall.bat first.
        """
        if platform.system() != "Windows":
            return cmd
        if not (settings.get("compile_dit") or settings.get("compile_vae")):
            return cmd

        vcvars_path = self._find_vcvars()
        if not vcvars_path:
            # Disable compile flags if VS Build Tools are not available
            self._log_lines.append("⚠️ VS Build Tools not found; disabling torch.compile for compatibility.")
            # Remove compile-related flags from command
            filtered_cmd = []
            skip_next = False
            for c in cmd:
                if skip_next:
                    skip_next = False
                    continue
                if c in ("--compile_dit", "--compile_vae", "--compile_backend", "--compile_mode",
                        "--compile_fullgraph", "--compile_dynamic"):
                    skip_next = True  # Skip the next argument too
                    continue
                if c.startswith("--compile_dynamo_cache_size_limit=") or c.startswith("--compile_dynamo_recompile_limit="):
                    continue
                filtered_cmd.append(c)
            return filtered_cmd

        # Build cmd /c call "vcvarsall.bat" x64 && original command
        quoted_cmd = " ".join(f'"{c}"' if " " in c else c for c in cmd)
        wrapped = ["cmd", "/c", f'call "{vcvars_path}" x64 && {quoted_cmd}']
        return wrapped

    def _maybe_clear_in_app_model(self, model_id: Optional[str]):
        """
        Best-effort memory trim when switching models in in-app mode.
        """
        if self._active_mode != "in_app":
            return
        if model_id is None:
            return
        if self._last_model_id and self._last_model_id != model_id:
            try:
                if torch:
                    torch.cuda.empty_cache()
            except Exception:
                pass
        self._last_model_id = model_id

    # ------------------------------------------------------------------ #
    # RIFE runner
    # ------------------------------------------------------------------ #
    def run_rife(
        self,
        settings: Dict[str, Any],
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> RunResult:
        """
        Run RIFE inference_video.py with given settings.
        """
        cli_path = self.base_dir / "RIFE" / "inference_video.py"
        if not cli_path.exists():
            raise FileNotFoundError(f"RIFE CLI not found at {cli_path}")

        input_path = normalize_path(settings.get("input_path"))
        if not input_path:
            raise ValueError("Input path is required for RIFE.")

        png_output = bool(settings.get("png_output"))
        output_override = settings.get("output_override") or None
        predicted_output = rife_output_path(
            input_path,
            png_output,
            output_override,
            global_output_dir=str(self.output_dir),
            png_padding=settings.get("png_padding"),
            png_keep_basename=settings.get("png_keep_basename", False),
        )

        cmd = self._build_rife_cmd(cli_path, input_path, predicted_output, settings)

        # In-app mode (not cancelable)
        if self._active_mode == "in_app":
            buf = io.StringIO()
            rc = 0
            try:
                with redirect_stdout(buf), redirect_stderr(buf):
                    sys.argv = cmd[1:]
                    runpy.run_path(str(cli_path), run_name="__main__")
            except SystemExit as exc:  # argparse exit
                rc = int(exc.code) if isinstance(exc.code, int) else 1
            except Exception as exc:
                buf.write(f"{exc}\n")
                rc = 1
            finally:
                sys.argv = [sys.executable]
            output_path = str(predicted_output)
            meta_payload = {
                "returncode": rc,
                "output": output_path,
                "args": settings,
            }
            if png_output:
                write_png_metadata(Path(output_path), meta_payload)
            else:
                emit_metadata(Path(output_path), meta_payload)
            return RunResult(rc, output_path, buf.getvalue())

        env = os.environ.copy()
        env.setdefault("PYTHONWARNINGS", "ignore")
        creationflags = 0
        preexec_fn = None
        if platform.system() == "Windows":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            preexec_fn = os.setsid  # type: ignore[arg-type]

        proc: Optional[subprocess.Popen] = None
        log_lines: List[str] = []
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                cwd=self.base_dir / "RIFE",
                creationflags=creationflags,
                preexec_fn=preexec_fn,
            )
            with self._lock:
                self._active_process = proc

            assert proc.stdout is not None
            for line in proc.stdout:
                log_lines.append(line.rstrip())
                if on_progress:
                    on_progress(line)
                with self._lock:
                    if self._active_process is None:
                        break
            proc.wait()
        finally:
            with self._lock:
                self._active_process = None

        output_path = str(predicted_output)
        if settings.get("fps_override") and predicted_output.suffix.lower() == ".mp4":
            adjusted = ffmpeg_set_fps(predicted_output, settings["fps_override"])
            output_path = str(adjusted)

        meta_payload = {
            "returncode": proc.returncode if proc else -1,
            "output": output_path,
            "args": settings,
        }
        if self._telemetry_enabled:
            if png_output:
                write_png_metadata(Path(output_path), meta_payload)
            else:
                emit_metadata(Path(output_path), meta_payload)

        return RunResult(proc.returncode if proc else -1, output_path, "\n".join(log_lines))

    def _build_rife_cmd(
        self,
        cli_path: Path,
        input_path: str,
        output_path: Path,
        settings: Dict[str, Any],
    ) -> List[str]:
        cmd: List[str] = [sys.executable, str(cli_path)]

        if settings.get("img_mode"):
            cmd.extend(["--img", input_path])
        else:
            cmd.extend(["--video", input_path])

        cmd.extend(["--output", str(output_path)])

        if settings.get("montage"):
            cmd.append("--montage")
        if settings.get("png_output"):
            cmd.append("--png")
        if settings.get("no_audio"):
            cmd.append("--no-audio")
        if settings.get("show_ffmpeg"):
            cmd.append("--show-ffmpeg")
        if settings.get("uhd_half"):
            cmd.append("--UHD")
        if settings.get("scale"):
            cmd.extend(["--scale", str(settings["scale"])])
        if settings.get("fps_multiplier"):
            cmd.extend(["--multi", str(settings["fps_multiplier"])])
        if settings.get("model_dir"):
            cmd.extend(["--model", settings["model_dir"]])
        if settings.get("cuda_device"):
            cmd.extend(["--cuda_device", str(settings["cuda_device"])])
        if settings.get("png_output"):
            cmd.extend(["--ext", "png"])

        return cmd

    # ------------------------------------------------------------------ #
    # Placeholder GAN/image-based upscaler runner (stub)
    # ------------------------------------------------------------------ #
    def run_gan_placeholder(
        self,
        settings: Dict[str, Any],
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> RunResult:
        """
        Stub runner for GAN/image-based upscalers.
        Emits a not-implemented message to keep UI responsive until pipeline is wired.
        """
        msg = "Image-based GAN upscaler is not yet wired to a backend. Configure pipeline to enable runs."
        if on_progress:
            on_progress(msg + "\n")
        return RunResult(returncode=1, output_path=None, log=msg)


