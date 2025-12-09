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
        """
        Cancel the active subprocess.
        
        Handles platform-specific termination:
        - Windows: Uses CTRL_BREAK_EVENT then terminate/kill
        - Unix: Uses SIGTERM then SIGKILL
        
        Returns True if cancellation was attempted, False if no active process.
        """
        with self._lock:
            proc = self._active_process
            self._canceled = True
        
        if not proc:
            return False
        
        try:
            if platform.system() == "Windows":
                # Windows-specific graceful shutdown
                try:
                    # First try CTRL_BREAK_EVENT (only works if CREATE_NEW_PROCESS_GROUP was used)
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                    
                    # Wait briefly for graceful shutdown
                    try:
                        proc.wait(timeout=2.0)
                        return True  # Process exited gracefully
                    except subprocess.TimeoutExpired:
                        pass
                except (OSError, AttributeError):
                    # CTRL_BREAK might not work, continue to terminate
                    pass
                
                # Try terminate
                try:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2.0)
                        return True
                    except subprocess.TimeoutExpired:
                        pass
                except OSError:
                    pass
                
                # Force kill as last resort
                try:
                    proc.kill()
                    proc.wait(timeout=1.0)
                except Exception:
                    pass
                    
            else:
                # Unix/Linux: SIGTERM then SIGKILL
                try:
                    proc.terminate()  # SIGTERM
                    try:
                        proc.wait(timeout=2.0)
                        return True
                    except subprocess.TimeoutExpired:
                        pass
                except OSError:
                    pass
                
                # Force kill
                try:
                    proc.kill()  # SIGKILL
                    proc.wait(timeout=1.0)
                except Exception:
                    pass
            
            return True
            
        except Exception as e:
            # Log error but don't crash
            print(f"Error during cancellation: {e}")
            return False
        finally:
            # Always clear the active process reference
            with self._lock:
                self._active_process = None
            
            # Clear CUDA cache after cancellation to free VRAM
            try:
                from .gpu_utils import clear_cuda_cache
                clear_cuda_cache()
                print("✅ CUDA cache cleared after cancellation")
            except Exception:
                # Silently ignore if CUDA not available
                pass
                
            # Clean up any zombie processes on Unix
            if platform.system() != "Windows":
                try:
                    import os
                    os.waitpid(-1, os.WNOHANG)
                except (ChildProcessError, OSError):
                    pass

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

        # Route based on execution mode
        if self._active_mode == "in_app":
            return self._run_seedvr2_in_app(cli_path, cmd, predicted_output, settings, on_progress)
        else:
            return self._run_seedvr2_subprocess(cli_path, cmd, predicted_output, settings, on_progress)

    def _run_seedvr2_subprocess(self, cli_path: Path, cmd: List[str], predicted_output: Optional[Path], settings: Dict[str, Any], on_progress: Optional[Callable[[str], None]] = None) -> RunResult:
        """
        Execute SeedVR2 CLI as a subprocess with proper error handling and logging.
        """
        # Log the command being executed for debugging
        cmd_str = " ".join(f'"{c}"' if " " in c else c for c in cmd)
        if on_progress:
            on_progress(f"Executing command: {cmd_str}\n")

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

        cmd = self._maybe_wrap_with_vcvars(cmd, settings)

        if on_progress:
            on_progress("Starting SeedVR2 subprocess (CLI will handle model loading)...\n")

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
        returncode = -1

        try:
            if on_progress:
                on_progress("Starting subprocess...\n")

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

            if on_progress:
                on_progress("Subprocess started, monitoring output...\n")

            assert proc.stdout is not None

            # Read output line by line with timeout handling
            while True:
                # Check if process is still running and not canceled
                with self._lock:
                    if self._active_process is None:
                        if on_progress:
                            on_progress("Process canceled by user\n")
                        break

                try:
                    # Use timeout to avoid blocking indefinitely
                    if platform.system() == "Windows":
                        # Windows doesn't have select for file handles, use simpler approach
                        line = proc.stdout.readline()
                        if not line:
                            break
                    else:
                        # Unix systems can use select
                        import select
                        ready, _, _ = select.select([proc.stdout], [], [], 1.0)
                        if ready:
                            line = proc.stdout.readline()
                            if not line:
                                break
                        else:
                            continue

                    line = line.rstrip()
                    if line:  # Only add non-empty lines
                        log_lines.append(line)
                        if on_progress:
                            on_progress(line + "\n")

                except Exception as e:
                    if on_progress:
                        on_progress(f"Error reading subprocess output: {e}\n")
                    break

            # Wait for process to complete
            returncode = proc.wait()

            if on_progress:
                on_progress(f"Subprocess completed with return code: {returncode}\n")
            
            # Clear CUDA cache after subprocess completes
            # This ensures VRAM is freed even if the subprocess didn't clean up properly
            try:
                from .gpu_utils import clear_cuda_cache
                clear_cuda_cache()
                if on_progress and returncode == 0:
                    on_progress("✅ CUDA cache cleared\n")
            except Exception:
                # Silently ignore if CUDA not available or clear fails
                pass

        except FileNotFoundError as e:
            error_msg = f"CLI script not found: {e}"
            log_lines.append(error_msg)
            if on_progress:
                on_progress(f"❌ {error_msg}\n")
            returncode = 1
        except Exception as e:
            error_msg = f"Failed to execute subprocess: {e}"
            log_lines.append(error_msg)
            if on_progress:
                on_progress(f"❌ {error_msg}\n")
            returncode = 1
        finally:
            with self._lock:
                self._active_process = None
            
            # Also clear CUDA cache on error/cancellation
            try:
                from .gpu_utils import clear_cuda_cache
                clear_cuda_cache()
            except Exception:
                pass

        # Determine output path
        output_path = None
        if predicted_output and Path(predicted_output).exists():
            output_path = str(predicted_output)
            if on_progress:
                on_progress(f"Output file created: {output_path}\n")
        elif predicted_output:
            if on_progress:
                on_progress(f"Expected output not found: {predicted_output}\n")

        # Handle cancellation case
        if self._canceled and predicted_output and Path(predicted_output).exists():
            log_lines.append("Run canceled; partial output preserved.")
            output_path = str(predicted_output)

        # Emit metadata if successful and enabled
        # Check both global telemetry AND per-run metadata settings
        should_emit_metadata = self._telemetry_enabled and settings.get("save_metadata", True)
        if output_path and returncode == 0 and should_emit_metadata:
            try:
                emit_metadata(
                    Path(output_path),
                    {
                        "returncode": returncode,
                        "output": output_path,
                        "args": settings,
                        "command": cmd_str,
                    },
                )
            except Exception as e:
                if on_progress:
                    on_progress(f"Warning: Failed to emit metadata: {e}\n")

        self._last_model_id = settings.get("dit_model", self._last_model_id)

        # Combine all log lines
        full_log = "\n".join(log_lines)

        return RunResult(returncode, output_path, full_log)

    def _run_seedvr2_in_app(self, cli_path: Path, cmd: List[str], predicted_output: Optional[Path], settings: Dict[str, Any], on_progress: Optional[Callable[[str], None]] = None) -> RunResult:
        """
        Execute SeedVR2 in-app mode (EXPERIMENTAL - Currently Not Implemented).

        CURRENT STATUS: This method is a stub that falls back to subprocess execution.
        
        PLANNED IMPLEMENTATION:
        True in-app mode will provide:
        - Direct model loading and caching (models stay in VRAM between runs)
        - Faster repeated processing (no model reload overhead)
        - Higher VRAM/RAM usage (models persist until app restart)
        - Non-cancelable during processing (no subprocess to kill)
        
        REQUIREMENTS FOR FULL IMPLEMENTATION:
        - Direct import of SeedVR2 inference modules (bypass CLI wrapper)
        - ModelManager integration for cross-run caching
        - Proper VRAM lifecycle management without subprocess cleanup
        - Manual model unloading controls in UI
        
        For now, this provides subprocess execution with the same guarantees:
        - Full VRAM cleanup after each run
        - Cancelable processing
        - No memory leaks
        """
        if on_progress:
            on_progress("ℹ️ In-app mode is currently experimental (stub implementation)\n")
            on_progress("ℹ️ Using subprocess mode for guaranteed VRAM cleanup and cancellation support\n")

        # Fall back to subprocess execution until in-app mode is fully implemented
        return self._run_seedvr2_subprocess(cli_path, cmd, predicted_output, settings, on_progress)

    def ensure_seedvr2_model_loaded(self, settings: Dict[str, Any], on_progress: Optional[Callable[[str], None]] = None) -> bool:
        """
        Ensure the required SeedVR2 model is loaded, loading it if necessary.
        
        Uses ModelManager for intelligent caching and delayed loading.
        In subprocess mode, the CLI will handle actual loading.
        In in-app mode, the ModelManager would cache the loaded model.
        """
        dit_model = settings.get("dit_model", "")
        if not dit_model:
            return False

        # In subprocess mode, CLI handles loading - but we still track state
        if self._active_mode == "subprocess":
            if on_progress:
                on_progress(f"Model '{dit_model}' will be loaded by subprocess CLI\n")
            
            # Update model manager state for tracking (even though CLI does the work)
            model_id = self._model_manager._generate_model_id(
                ModelType.SEEDVR2,
                dit_model,
                **settings
            )
            self._model_manager.current_model_id = model_id
            
            return True
        
        # In in-app mode, use ModelManager for actual caching
        # (This would require implementing in-app model loading, which is future work)
        # For now, always defer to CLI
        if on_progress:
            on_progress(f"Model '{dit_model}' loading delegated to CLI\n")
        
        return True

    # ------------------------------------------------------------------ #
    # Command builder
    # ------------------------------------------------------------------ #
    def _find_vcvars(self) -> Optional[Path]:
        """
        Try to locate vcvarsall.bat using multiple detection methods.
        
        Uses the most robust approach:
        1. Check VSINSTALLDIR environment variable
        2. Use vswhere.exe (official VS installer locator)
        3. Fall back to hardcoded common paths
        
        This ensures maximum compatibility across different VS installations.
        """
        # Method 1: Check environment variable (fastest)
        vs_install_dir = os.environ.get("VSINSTALLDIR")
        if vs_install_dir:
            env_candidate = Path(vs_install_dir) / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
            if env_candidate.exists():
                return env_candidate
        
        # Method 2: Use vswhere.exe (most reliable)
        vswhere = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
        if vswhere.exists():
            try:
                # Query vswhere for latest Visual Studio installation
                result = subprocess.run(
                    [str(vswhere), "-latest", "-property", "installationPath"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0 and result.stdout.strip():
                    install_path = Path(result.stdout.strip())
                    vcvars = install_path / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
                    if vcvars.exists():
                        return vcvars
            except (subprocess.TimeoutExpired, Exception):
                # vswhere failed, continue to fallback
                pass
        
        # Method 3: Check common installation paths
        candidates = [
            # BuildTools (most common for CI/CD)
            Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"),
            Path(r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"),
            # Community Edition
            Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"),
            Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"),
            # Professional Edition
            Path(r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat"),
            # Enterprise Edition
            Path(r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat"),
            # VS 2019 fallback
            Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"),
            Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat"),
        ]
        
        for candidate in candidates:
            if candidate.exists():
                # Quick validation that it's actually a vcvarsall.bat file
                try:
                    with open(candidate, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read(200)
                        # Verify it looks like a vcvarsall.bat file
                        if 'vcvarsall' in content.lower() or '@echo off' in content:
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
        _add_int("--chunk_size", "chunk_size")  # SeedVR2 native streaming mode
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
        # Check both global telemetry AND per-run metadata settings
        should_emit_metadata = self._telemetry_enabled and settings.get("save_metadata", True)
        if should_emit_metadata:
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

        # Model and device settings
        if settings.get("model_dir"):
            cmd.extend(["--model", settings["model_dir"]])
        if settings.get("fp16_mode"):
            cmd.append("--fp16")
        if settings.get("uhd_mode"):
            cmd.append("--UHD")

        # Processing parameters
        if settings.get("scale") and settings["scale"] != 1.0:
            cmd.extend(["--scale", str(settings["scale"])])
        if settings.get("fps_multiplier") and settings["fps_multiplier"] != 2:
            cmd.extend(["--multi", str(int(settings["fps_multiplier"]))])
        if settings.get("fps_override") and settings["fps_override"] > 0:
            cmd.extend(["--fps", str(settings["fps_override"])])
        if settings.get("exp") and settings["exp"] != 1:
            cmd.extend(["--exp", str(settings["exp"])])

        # Output options
        if settings.get("png_output"):
            cmd.append("--png")
        if settings.get("montage"):
            cmd.append("--montage")
        if settings.get("no_audio"):
            cmd.append("--no-audio")
        if settings.get("show_ffmpeg"):
            cmd.append("--show-ffmpeg")
        if settings.get("skip_static_frames"):
            cmd.append("--skip")

        # Frame control
        if settings.get("skip_first_frames") and settings["skip_first_frames"] > 0:
            # RIFE uses --skip for static frames, not for frame skipping
            pass
        if settings.get("load_cap") and settings["load_cap"] > 0:
            # RIFE doesn't have a direct load_cap equivalent
            pass

        return cmd

    # ------------------------------------------------------------------ #
    # GAN/image-based upscaler runner
    # ------------------------------------------------------------------ #
    def run_gan(
        self,
        settings: Dict[str, Any],
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> RunResult:
        """
        Run GAN-based upscaling using the proper GAN runner implementation.
        """
        from .gan_runner import run_gan_upscale, GanResult

        try:
            # Convert settings to GAN runner format
            model_name = settings.get("model", "")
            if not model_name:
                error_msg = "No GAN model specified"
                if on_progress:
                    on_progress(f"❌ {error_msg}\n")
                return RunResult(1, None, error_msg)

            input_path = normalize_path(settings.get("input_path"))
            if not input_path or not Path(input_path).exists():
                error_msg = f"Input path not found: {input_path}"
                if on_progress:
                    on_progress(f"❌ {error_msg}\n")
                return RunResult(1, None, error_msg)

            if on_progress:
                on_progress(f"Starting GAN upscaling with model: {model_name}\n")
                on_progress(f"Input: {input_path}\n")

            # Run the GAN processing using proper runner
            result: GanResult = run_gan_upscale(
                input_path=input_path,
                model_name=model_name,
                settings=settings,
                base_dir=self.base_dir,
                temp_dir=self.temp_dir,
                output_dir=self.output_dir,
                on_progress=on_progress,
                cancel_event=None
            )

            # Output path comes from result
            output_path = result.output_path

            # Emit metadata if successful and enabled
            should_emit_metadata = self._telemetry_enabled and settings.get("save_metadata", True)
            if output_path and result.returncode == 0 and should_emit_metadata:
                try:
                    emit_metadata(
                        Path(output_path),
                        {
                            "returncode": result.returncode,
                            "output": output_path,
                            "args": settings,
                            "model": model_name,
                        },
                    )
                except Exception as e:
                    if on_progress:
                        on_progress(f"Warning: Failed to emit metadata: {e}\n")

            return RunResult(result.returncode, output_path, result.log)

        except Exception as e:
            error_msg = f"GAN processing failed: {str(e)}"
            if on_progress:
                on_progress(f"❌ {error_msg}\n")
            return RunResult(1, None, error_msg)


