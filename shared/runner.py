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
    get_media_fps,
    normalize_path,
    resolve_output_location,
    rife_output_path,
    write_png_metadata,
    detect_input_type,
)
from .model_manager import get_model_manager, ModelType
from .command_logger import get_command_logger

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
        
        # Initialize command logger
        executed_commands_dir = self.base_dir.parent / "executed_commands"
        self._command_logger = get_command_logger(executed_commands_dir)

    # ------------------------------------------------------------------ #
    # Mode management
    # ------------------------------------------------------------------ #
    def set_mode(self, mode: str):
        """
        Set execution mode: 'subprocess' or 'in_app'.
        
        MODEL-SPECIFIC BEHAVIOR:
        - SeedVR2: Always uses subprocess for CLI execution (even in 'in_app' mode).
                   CLI architecture prevents persistent model caching.
        - GAN: Can benefit from in-app mode (models can persist between runs).
        - RIFE: Can benefit from in-app mode (models can persist between runs).
        - FlashVSR+: Similar to SeedVR2, CLI-based (limited in-app benefit).
        
        IMPORTANT: In-app mode has NO cancellation support and requires manual
        vcvars activation on Windows for torch.compile. Subprocess mode is
        STRONGLY RECOMMENDED for all use cases.
        """
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

                # EXTRA SAFETY: kill the whole process tree (SeedVR2 may spawn helpers).
                # Prevents orphaned child processes from holding VRAM/handles after cancel.
                try:
                    subprocess.run(
                        ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
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
                # Note: If the *main* app process has initialized CUDA, Windows may still
                # show a small persistent VRAM reservation (CUDA context overhead).
                # That is expected and only fully disappears when the app exits.
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
        # FIXED: Always honor global output_dir, even when Output Override is empty
        # The SeedVR2 CLI's generate_output_path() REQUIRES --output to use custom directory,
        # otherwise it defaults to writing next to input. We must always pass --output.
        effective_output_override = settings.get("output_override") or None
        if preview_only:
            # Force single-frame preview: load_cap=1, image path remains same
            settings = settings.copy()
            settings["load_cap"] = 1
            # The CLI supports batch_size=1 (4n+1 rule) and recommends it for 1-frame runs.
            # Our UI batch size slider starts at 5, so force a sane preview value here.
            try:
                settings["batch_size"] = 1
                settings["uniform_batch_size"] = False
            except Exception:
                pass

        # Respect CLI auto-detect: if format is None/auto, choose based on input type
        if preview_only:
            effective_output_format = "png"
        elif format_for_cli is None:
            itype = detect_input_type(input_path)
            effective_output_format = "png" if itype in ("image", "directory") else "mp4"
        else:
            effective_output_format = format_for_cli

        # FIXED: Predict output path AND ensure CLI receives --output for global output_dir
        # Even when user doesn't set Output Override, we need to pass global output_dir to CLI
        predicted_output: Optional[Path]
        cli_output_arg: Optional[str] = None  # What we pass to CLI via --output
        
        if effective_output_override:
            # User explicitly set Output Override - use it
            override_path = Path(normalize_path(effective_output_override))
            if override_path.suffix:
                # Explicit file path
                predicted_output = override_path
                cli_output_arg = str(override_path)
            else:
                # Directory override
                cli_output_arg = str(override_path)
                predicted_output = resolve_output_location(
                    input_path=input_path,
                    output_format=effective_output_format,
                    global_output_dir=cli_output_arg,
                    batch_mode=batch_mode,
                    png_padding=settings.get("png_padding"),
                    png_keep_basename=settings.get("png_keep_basename", False),
                    original_filename=settings.get("_original_filename"),  # Preserve user's filename
                )
        else:
            # FIXED: No override, but still pass global output_dir to CLI
            # This ensures files go to user's configured output folder, not next to input
            cli_output_arg = str(self.output_dir)  # CRITICAL: Pass to CLI even when no override
            predicted_output = resolve_output_location(
                input_path=input_path,
                output_format=effective_output_format,
                global_output_dir=cli_output_arg,
                batch_mode=batch_mode,
                png_padding=settings.get("png_padding"),
                png_keep_basename=settings.get("png_keep_basename", False),
                original_filename=settings.get("_original_filename"),  # Preserve user's filename
            )

        # FIXED: Pass cli_output_arg to command builder (not effective_output_override)
        cmd = self._build_seedvr2_cmd(cli_path, settings, format_for_cli, preview_only, output_override=cli_output_arg)

        # Route based on execution mode
        if self._active_mode == "in_app":
            return self._run_seedvr2_in_app(cli_path, cmd, predicted_output, settings, on_progress)
        else:
            result = self._run_seedvr2_subprocess(cli_path, cmd, predicted_output, settings, on_progress)

            # -----------------------------------------------------------------
            # Windows hard-crash auto-retry
            # -----------------------------------------------------------------
            # Some CUDA extensions (flash-attn / sage-attn / triton) can hard-crash
            # on Windows with 0xC0000005 (3221225477) depending on GPU + build.
            # We can't catch a Python exception because the process dies, so we
            # provide a single safe retry using PyTorch SDPA.
            windows_access_violation = (platform.system() == "Windows" and result.returncode == 3221225477)
            current_attn = str(settings.get("attention_mode") or "").strip().lower()
            if windows_access_violation and current_attn and current_attn != "sdpa":
                retry_msg = (
                    "\n[SeedVR2] ⚠️ Detected Windows native crash (0xC0000005 / access violation).\n"
                    "[SeedVR2] ↻ Auto-retrying with safer settings: attention_mode=sdpa"
                )
                if preview_only or int(settings.get("load_cap") or 0) == 1:
                    retry_msg += ", batch_size=1"
                retry_msg += "\n"

                print(retry_msg, flush=True)
                if on_progress:
                    on_progress(retry_msg)

                retry_settings = settings.copy()
                retry_settings["attention_mode"] = "sdpa"
                # Ensure compile stays off for the retry (compile-related crashes can also happen)
                retry_settings["compile_dit"] = False
                retry_settings["compile_vae"] = False

                # Safe batch size for tiny runs (incl. preview-only)
                if preview_only or int(retry_settings.get("load_cap") or 0) == 1:
                    retry_settings["batch_size"] = 1
                    retry_settings["uniform_batch_size"] = False

                retry_cmd = self._build_seedvr2_cmd(
                    cli_path,
                    retry_settings,
                    format_for_cli,
                    preview_only,
                    output_override=cli_output_arg,
                )
                retry_result = self._run_seedvr2_subprocess(
                    cli_path, retry_cmd, predicted_output, retry_settings, on_progress
                )

                combined_log = "\n".join(
                    [
                        "=== SeedVR2 attempt 1 (original settings) ===",
                        result.log or "",
                        "",
                        "=== SeedVR2 attempt 2 (auto-retry: attention_mode=sdpa) ===",
                        retry_result.log or "",
                    ]
                )
                return RunResult(retry_result.returncode, retry_result.output_path, combined_log)

            return result

    def _run_seedvr2_subprocess(self, cli_path: Path, cmd: List[str], predicted_output: Optional[Path], settings: Dict[str, Any], on_progress: Optional[Callable[[str], None]] = None) -> RunResult:
        """
        Execute SeedVR2 CLI as a subprocess with proper error handling and logging.
        
        ENHANCED: Now prints all subprocess output to console (CMD) for user visibility,
        in addition to sending it to the on_progress callback for UI updates.
        """
        # Helper function to log to both console AND callback
        def log_output(message: str, force_console: bool = True):
            """Log message to console (always) and callback (if provided)."""
            if force_console:
                print(message, end='', flush=True)  # Print to CMD for user visibility
            if on_progress:
                on_progress(message)
        
        # Visual separator for user visibility in CMD
        print("\n[SeedVR2] Starting upscaling process...", flush=True)
        
        # Show key settings for user visibility
        input_path = settings.get("input_path", "Unknown")
        model_name = settings.get("dit_model", "Unknown")
        resolution = settings.get("resolution", "Unknown")
        print(f"[SeedVR2] Input: {input_path}", flush=True)
        print(f"[SeedVR2] Model: {model_name}", flush=True)
        print(f"[SeedVR2] Resolution: {resolution}p", flush=True)
        
        # Log the command being executed for debugging
        cmd_str = " ".join(f'"{c}"' if " " in c else c for c in cmd)
        log_output(f"[SeedVR2] Executing command:\n{cmd_str}\n")

        # FIXED: Pass on_progress to _maybe_wrap_with_vcvars for transparent warning surfacing
        # vcvars wrapper now handles compile detection and warning display
        cmd = self._maybe_wrap_with_vcvars(cmd, settings, on_progress)

        log_output("[SeedVR2] Starting subprocess (CLI will handle model loading)...\n")

        env = os.environ.copy()
        env["TEMP"] = str(self.temp_dir)
        env["TMP"] = str(self.temp_dir)
        env.setdefault("PYTHONWARNINGS", "ignore")
        # Windows consoles often default to legacy code pages (cp1252) which can crash
        # SeedVR2 when it prints emojis (UnicodeEncodeError). Force UTF-8 for the CLI.
        if platform.system() == "Windows":
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"

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
        start_time = time.time()  # Track execution time

        try:
            log_output("[SeedVR2] Launching CLI process...\n")

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                cwd=self.base_dir,
                creationflags=creationflags,
                preexec_fn=preexec_fn,
            )

            with self._lock:
                self._active_process = proc
                self._canceled = False

            log_output("[SeedVR2] Process started, monitoring output...\n")

            assert proc.stdout is not None

            # Read output line by line with timeout handling
            while True:
                # Check if process is still running and not canceled
                with self._lock:
                    if self._active_process is None:
                        log_output("[SeedVR2] Process canceled by user\n")
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
                        # Print to console for user visibility AND send to callback
                        print(line, flush=True)  # Always print to CMD
                        if on_progress:
                            on_progress(line + "\n")

                except Exception as e:
                    log_output(f"[SeedVR2] Error reading subprocess output: {e}\n")
                    break

            # Wait for process to complete
            returncode = proc.wait()

            # Log completion status with clear visual indicator
            if returncode == 0:
                print("[SeedVR2] Process completed successfully", flush=True)
                log_output(f"[SeedVR2] Process completed successfully (code: {returncode})\n")
            else:
                print(f"[SeedVR2] Process exited with error code: {returncode}", flush=True)
                log_output(f"[SeedVR2] Process exited with error code: {returncode}\n")
                
                # Show last few log lines as error context
                if log_lines:
                    print("\n[SeedVR2] Last 15 lines of output:", flush=True)
                    for line in log_lines[-15:]:
                        print(f"  {line}", flush=True)
                    log_output("[SeedVR2] Last 15 lines of output:\n")
                    for line in log_lines[-15:]:
                        log_output(f"  {line}\n")
                else:
                    print("\n[SeedVR2] No output captured from subprocess!", flush=True)
                    print("Possible causes:", flush=True)
                    print("  1. Python/CUDA initialization failed", flush=True)
                    print("  2. Missing dependencies or models", flush=True)
                    print("  3. Invalid input file format", flush=True)
                    print("  4. Insufficient VRAM or memory", flush=True)
                    log_output("[SeedVR2] No output captured from subprocess - check CMD for details\n")
            
            # Clear CUDA cache after subprocess completes
            # This ensures VRAM is freed even if the subprocess didn't clean up properly
            try:
                from .gpu_utils import clear_cuda_cache
                clear_cuda_cache()
                if returncode == 0:
                    log_output("[SeedVR2] CUDA cache cleared\n")
            except Exception:
                # Silently ignore if CUDA not available or clear fails
                pass

        except FileNotFoundError as e:
            error_msg = f"CLI script not found: {e}"
            log_lines.append(error_msg)
            print(f"[SeedVR2] ERROR: FILE NOT FOUND", flush=True)
            print(f"[SeedVR2] CLI Path: {cli_path}", flush=True)
            print(f"[SeedVR2] Error: {e}", flush=True)
            log_output(f"[SeedVR2] {error_msg}\n")
            returncode = 1
        except Exception as e:
            error_msg = f"Failed to execute subprocess: {e}"
            log_lines.append(error_msg)
            print(f"[SeedVR2] ERROR: SUBPROCESS EXECUTION FAILED", flush=True)
            print(f"[SeedVR2] Error Type: {type(e).__name__}", flush=True)
            print(f"[SeedVR2] Error Message: {e}", flush=True)
            log_output(f"[SeedVR2] {error_msg}\n")
            # Print full traceback to CMD for debugging
            import traceback
            traceback_str = traceback.format_exc()
            print("\n[SeedVR2] FULL TRACEBACK:", flush=True)
            print(traceback_str, flush=True)
            log_lines.append(traceback_str)
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
            log_output(f"[SeedVR2] Output file created: {output_path}\n")
        elif predicted_output:
            log_output(f"[SeedVR2] ⚠️ Expected output not found: {predicted_output}\n")

        # Handle cancellation case
        if self._canceled and predicted_output and Path(predicted_output).exists():
            log_lines.append("Run canceled; partial output preserved.")
            output_path = str(predicted_output)

        # Emit metadata for ALL runs (success, failure, cancellation) if enabled
        # This provides crucial telemetry for troubleshooting failed/cancelled runs
        # Check both global telemetry AND per-run metadata settings
        should_emit_metadata = self._telemetry_enabled and settings.get("save_metadata", True)
        if should_emit_metadata:
            try:
                # Determine metadata target: output path if exists, otherwise output_dir
                metadata_target = Path(output_path) if output_path else self.output_dir
                
                # Build comprehensive metadata including failure/cancellation info
                metadata_payload = {
                    "returncode": returncode,
                    "output": output_path,
                    "args": settings,
                    "command": cmd_str,
                    "status": "success" if returncode == 0 else ("cancelled" if self._canceled else "failed"),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                
                # Add failure-specific context
                if returncode != 0:
                    metadata_payload["error_logs"] = log_lines[-50:]  # Last 50 log lines for debugging
                    if self._canceled:
                        metadata_payload["cancellation_reason"] = "User cancelled processing"
                
                emit_metadata(metadata_target, metadata_payload)
            except Exception as e:
                if on_progress:
                    on_progress(f"Warning: Failed to emit metadata: {e}\n")

        self._last_model_id = settings.get("dit_model", self._last_model_id)

        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Log command to executed_commands folder
        try:
            self._command_logger.log_command(
                tab_name="seedvr2",
                command=cmd,
                settings=settings,
                returncode=returncode,
                output_path=output_path,
                error_logs=log_lines[-50:] if returncode != 0 else None,  # Last 50 lines on error
                execution_time=execution_time,
                additional_info={
                    "mode": "subprocess",
                    "cancelled": self._canceled,
                    "predicted_output": str(predicted_output) if predicted_output else None
                }
            )
            log_output(f"[SeedVR2] ✅ Command logged to executed_commands folder\n")
        except Exception as e:
            log_output(f"[SeedVR2] ⚠️ Failed to log command: {e}\n")

        # Combine all log lines
        full_log = "\n".join(log_lines)

        return RunResult(returncode, output_path, full_log)

    def _run_seedvr2_in_app(self, cli_path: Path, cmd: List[str], predicted_output: Optional[Path], settings: Dict[str, Any], on_progress: Optional[Callable[[str], None]] = None) -> RunResult:
        """
        Execute SeedVR2 in-app mode (EXPERIMENTAL - NOT RECOMMENDED).

        ⚠️ CRITICAL LIMITATION: SeedVR2 CLI ARCHITECTURE PREVENTS MODEL PERSISTENCE
        ============================================================================
        The SeedVR2 CLI is designed to load models, process, then exit. Even when run
        via runpy (in-process), the CLI code does NOT maintain model instances between
        runs. Each invocation reloads models from disk.
        
        RESULT: In-app mode provides **ZERO SPEED BENEFIT** for SeedVR2 compared to subprocess.
        
        CURRENT IMPLEMENTATION STATUS:
        - ⚠️ PARTIALLY IMPLEMENTED: Runs CLI via runpy but does NOT implement persistent model caching
        - ❌ Models reload each run (IDENTICAL to subprocess mode - no performance gain)
        - ❌ Cannot cancel mid-run (no subprocess to kill)
        - ❌ VS Build Tools wrapper not applied (torch.compile may fail on Windows)
        - ⚠️ Memory leaks possible without subprocess isolation
        - ⚠️ ModelManager tracking exists but cannot force CLI to keep models loaded
        
        WHY THIS EXISTS:
        - Framework placeholder for future GAN/RIFE in-app optimization
        - Demonstrates in-app execution pattern for other model types
        - SeedVR2 would need CLI refactoring to support true model persistence
        
        RECOMMENDATION FOR SEEDVR2:
        🚫 **DO NOT USE IN-APP MODE** - It provides no benefits and loses cancellation.
        ✅ **USE SUBPROCESS MODE** - Same speed, full cancellation, better isolation.
        
        FUTURE WORK (requires SeedVR2 CLI changes):
        - Refactor CLI to expose model loading/inference as separate functions
        - Implement persistent model caching in ModelManager
        - Add intelligent model swapping when user changes models
        - Enable proper cancellation via threading interrupts
        """
        if on_progress:
            on_progress("⚠️ IN-APP MODE ACTIVE (NOT RECOMMENDED FOR SEEDVR2)\n")
            on_progress("🚫 CRITICAL: SeedVR2 CLI reloads models each run - NO SPEED BENEFIT over subprocess\n")
            on_progress("❌ LIMITATION: Cannot cancel mid-run (no subprocess to kill)\n")
            on_progress("💡 RECOMMENDATION: Use subprocess mode for SeedVR2 (same speed + cancellation)\n")
        
        # Check for compile + Windows - attempt vcvars environment setup
        if platform.system() == "Windows" and (settings.get("compile_dit") or settings.get("compile_vae")):
            from .health import is_vs_build_tools_available
            
            # Check if vcvars environment is already active
            vcvars_active = os.environ.get("VSCMD_ARG_TGT_ARCH") is not None
            
            if not vcvars_active:
                if on_progress:
                    on_progress("⚠️ WARNING: torch.compile requested but vcvars environment not active.\n")
                
                # Try to find and source vcvars
                vcvars_path = self._find_vcvars()
                
                if vcvars_path and vcvars_path.exists():
                    if on_progress:
                        on_progress(f"🔧 Attempting to activate VS Build Tools: {vcvars_path}\n")
                    
                    # In-app mode limitation: We cannot directly modify the current process environment
                    # after Python has started. The vcvars.bat sets up C++ compiler paths, but these
                    # need to be active BEFORE Python imports torch.
                    if on_progress:
                        on_progress("⚠️ IN-APP LIMITATION: Cannot activate vcvars after Python started.\n")
                        on_progress("💡 WORKAROUND: Activate vcvars BEFORE starting this app, or use subprocess mode.\n")
                        on_progress("🚫 Auto-disabling torch.compile to prevent cryptic compilation errors.\n")
                    
                    settings["compile_dit"] = False
                    settings["compile_vae"] = False
                else:
                    if on_progress:
                        on_progress("❌ VS Build Tools not found. torch.compile disabled.\n")
                        on_progress("💡 Install 'Desktop development with C++' workload from Visual Studio Installer.\n")
                    
                    settings["compile_dit"] = False
                    settings["compile_vae"] = False
            else:
                if on_progress:
                    on_progress("✅ VS Build Tools environment active - torch.compile should work.\n")

        log_lines: List[str] = []
        returncode = -1
        
        try:
            # Prepare environment
            env = os.environ.copy()
            env["TEMP"] = str(self.temp_dir)
            env["TMP"] = str(self.temp_dir)
            env.setdefault("PYTHONWARNINGS", "ignore")
            
            # Ensure directories exist
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            if on_progress:
                on_progress("Loading model and initializing pipeline...\n")
            
            # Track model loading via ModelManager
            model_manager = self._model_manager
            dit_model = settings.get("dit_model", "")
            model_id = model_manager._generate_model_id(ModelType.SEEDVR2, dit_model, **settings)
            
            # Update current model tracking
            old_model_id = model_manager.current_model_id
            if old_model_id and old_model_id != model_id:
                if on_progress:
                    on_progress(f"Model changed ({old_model_id} → {model_id}), clearing cache...\n")
                try:
                    from .gpu_utils import clear_cuda_cache
                    clear_cuda_cache()
                except Exception:
                    pass
            
            model_manager.current_model_id = model_id
            
            # Run CLI directly via runpy (stays in same process, models persist)
            # Build sys.argv from cmd (skip python executable)
            import sys
            old_argv = sys.argv.copy()
            
            try:
                # cmd format: [python_path, cli_path, ...args]
                sys.argv = [str(cli_path)] + cmd[2:]  # Skip python path and cli path
                
                if on_progress:
                    on_progress(f"Executing: {' '.join(sys.argv)}\n")
                
                # Capture output
                log_buffer = io.StringIO()
                with redirect_stdout(log_buffer), redirect_stderr(log_buffer):
                    try:
                        # Run the CLI script directly in this process
                        runpy.run_path(str(cli_path), run_name="__main__")
                        returncode = 0
                    except SystemExit as e:
                        returncode = e.code if isinstance(e.code, int) else (1 if e.code else 0)
                    except Exception as e:
                        log_lines.append(f"❌ In-app execution error: {str(e)}")
                        returncode = 1
                
                log_lines.append(log_buffer.getvalue())
                
            finally:
                # Restore sys.argv
                sys.argv = old_argv
            
            if on_progress:
                on_progress(f"Execution completed with code {returncode}\n")
            
            # Check for output
            output_path = None
            if predicted_output and predicted_output.exists():
                output_path = str(predicted_output)
                if on_progress:
                    on_progress(f"Output created: {output_path}\n")
            
            # Emit metadata for ALL runs (success, failure, cancellation) if enabled
            should_emit_metadata = self._telemetry_enabled and settings.get("save_metadata", True)
            if should_emit_metadata:
                try:
                    # Determine metadata target
                    metadata_target = Path(output_path) if output_path else self.output_dir
                    
                    metadata_payload = {
                        "returncode": returncode,
                        "output": output_path,
                        "args": settings,
                        "mode": "in_app",
                        "command": " ".join(cmd),
                        "status": "success" if returncode == 0 else "failed",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    
                    # Add failure context
                    if returncode != 0:
                        metadata_payload["error_logs"] = log_lines[-50:]  # Last 50 lines for debugging
                    
                    emit_metadata(metadata_target, metadata_payload)
                except Exception as e:
                    if on_progress:
                        on_progress(f"Warning: Failed to emit metadata: {e}\n")
            
            return RunResult(returncode, output_path, "\n".join(log_lines))
            
        except Exception as e:
            error_msg = f"In-app execution failed: {str(e)}"
            log_lines.append(f"❌ {error_msg}")
            if on_progress:
                on_progress(f"{error_msg}\n")
            return RunResult(1, None, "\n".join(log_lines))

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

    def _get_python_executable(self) -> str:
        """
        Get the correct Python executable path.
        
        Priority:
        1. Venv python from launcher (venv/Scripts/python.exe or venv/bin/python)
        2. sys.executable as fallback
        
        This ensures subprocess runs use the correct environment with all dependencies.
        """
        # Try to find venv python relative to base_dir
        if platform.system() == "Windows":
            venv_python = self.base_dir / "venv" / "Scripts" / "python.exe"
        else:
            venv_python = self.base_dir / "venv" / "bin" / "python"
        
        if venv_python.exists():
            return str(venv_python)
        
        # Fallback to sys.executable if venv not found
        return sys.executable

    def _build_seedvr2_cmd(
        self,
        cli_path: Path,
        settings: Dict[str, Any],
        output_format: Optional[str],
        preview_only: bool,
        output_override: Optional[str],
    ) -> List[str]:
        cmd: List[str] = [self._get_python_executable(), str(cli_path), settings["input_path"]]

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
        
        # ADDED v2.5.22: FFmpeg 10-bit encoding support
        # Video backend selection (opencv or ffmpeg)
        if settings.get("video_backend"):
            backend = str(settings["video_backend"]).lower()
            if backend in ("opencv", "ffmpeg"):
                cmd.extend(["--video_backend", backend])
        
        # 10-bit color depth (requires ffmpeg backend)
        if settings.get("use_10bit"):
            cmd.append("--10bit")

        # Preview: prefer PNG for quick visualization
        if preview_only and not output_format:
            cmd.extend(["--output_format", "png"])

        return cmd

    def _maybe_wrap_with_vcvars(self, cmd: List[str], settings: Dict[str, Any], on_progress: Optional[Callable[[str], None]] = None) -> List[str]:
        """
        On Windows, wrap command with vcvarsall.bat to activate C++ toolchain.
        
        The C++ toolchain is required for:
        - torch.compile (when compile_dit/compile_vae are enabled)
        - Some CUDA operations that need nvcc at runtime
        
        Behavior:
        - If compile flags are set: REQUIRE vcvars, disable compile if missing
        - If no compile flags: OPTIONALLY wrap with vcvars if available (best-effort C++ support)
        
        FIXED: Now surfaces warnings to UI via on_progress callback for transparency
        """
        if platform.system() != "Windows":
            return cmd
        
        compile_requested = settings.get("compile_dit") or settings.get("compile_vae")
        vcvars_path = self._find_vcvars()
        
        if not vcvars_path:
            if compile_requested:
                # Compile was requested but vcvars not found - disable compile flags
                warning_msg = "⚠️ VS Build Tools not found; disabling torch.compile for compatibility.\n" \
                             "Install 'Desktop development with C++' workload from Visual Studio Installer for torch.compile support.\n"
                self._log_lines.append(warning_msg.strip())
                
                # FIXED: Surface warning to UI so user knows compile was disabled
                if on_progress:
                    on_progress(warning_msg)
                
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
            else:
                # No compile requested and vcvars not found - proceed without vcvars
                # Log a warning but don't block execution
                if not hasattr(self, '_vcvars_warning_shown'):
                    info_msg = "ℹ️ VS Build Tools not found. torch.compile will be unavailable.\n"
                    self._log_lines.append(info_msg.strip())
                    if on_progress:
                        on_progress(info_msg)
                    self._vcvars_warning_shown = True
                return cmd
        
        # FIXED: Only wrap with vcvars if compile is actually requested
        # Wrapping with cmd /c can cause subprocess output capture issues
        if not compile_requested:
            # No compile requested - don't wrap with vcvars (avoid output capture issues)
            if on_progress:
                on_progress("ℹ️ VS Build Tools available but torch.compile not enabled\n")
            return cmd
        
        # Compile requested and vcvars found - wrap command to enable C++ toolchain
        if on_progress:
            on_progress(f"✅ Using VS Build Tools for torch.compile: {vcvars_path}\n")
        quoted_cmd = " ".join(f'"{c}"' if " " in c else c for c in cmd)
        wrapped = ["cmd", "/c", f'call "{vcvars_path}" x64 >nul 2>&1 && {quoted_cmd}']
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
        
        FIXED: Now supports skip_first_frames and load_cap via ffmpeg preprocessing.
        RIFE CLI doesn't have these options natively, so we pre-trim the video.
        """
        cli_path = self.base_dir / "RIFE" / "inference_video.py"
        if not cli_path.exists():
            raise FileNotFoundError(f"RIFE CLI not found at {cli_path}")

        input_path = normalize_path(settings.get("input_path"))
        if not input_path:
            raise ValueError("Input path is required for RIFE.")

        # FIXED: Pre-process video if skip_first_frames or load_cap is set
        # RIFE CLI doesn't support these natively, so we trim via ffmpeg first
        skip_frames = int(settings.get("skip_first_frames", 0))
        load_cap = int(settings.get("load_cap", 0))
        
        effective_input = input_path
        trimmed_video = None
        
        if (skip_frames > 0 or load_cap > 0) and detect_input_type(input_path) == "video":
            # Need to trim video via ffmpeg
            trimmed_video = self.temp_dir / f"rife_trimmed_{Path(input_path).stem}.mp4"
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            
            if on_progress:
                on_progress(f"Pre-trimming video (skip {skip_frames}, cap {load_cap})...\n")
            
            # Build ffmpeg trim command
            trim_cmd = ["ffmpeg", "-y", "-i", input_path]
            
            if skip_frames > 0:
                # Get FPS to convert frames to seconds
                fps = get_media_fps(input_path) or 30.0
                start_time = skip_frames / fps
                trim_cmd.extend(["-ss", str(start_time)])
            
            if load_cap > 0:
                # Limit number of frames
                trim_cmd.extend(["-frames:v", str(load_cap)])
            
            # Copy codec for fast trim (no re-encode)
            trim_cmd.extend(["-c", "copy", str(trimmed_video)])
            
            try:
                subprocess.run(trim_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                effective_input = str(trimmed_video)
                
                if on_progress:
                    on_progress(f"✅ Video trimmed: {trimmed_video.name}\n")
            except Exception as e:
                if on_progress:
                    on_progress(f"⚠️ Video trimming failed: {e}, using original input\n")
                # Fall back to original if trim fails
                effective_input = input_path

        png_output = bool(settings.get("png_output"))
        output_override = settings.get("output_override") or None
        predicted_output = rife_output_path(
            effective_input,  # Use trimmed input for output naming
            png_output,
            output_override,
            global_output_dir=str(self.output_dir),
            png_padding=settings.get("png_padding"),
            png_keep_basename=settings.get("png_keep_basename", False),
        )

        cmd = self._build_rife_cmd(cli_path, effective_input, predicted_output, settings)
        
        # Wrap with vcvars for C++ toolchain support (Windows only, best-effort)
        # FIXED: Pass on_progress for transparent warning surfacing
        if self._active_mode == "subprocess":
            cmd = self._maybe_wrap_with_vcvars(cmd, settings, on_progress)

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
        start_time = time.time()  # Track execution time
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

        returncode_val = proc.returncode if proc else -1
        execution_time = time.time() - start_time
        
        meta_payload = {
            "returncode": returncode_val,
            "output": output_path,
            "args": settings,
            "status": "success" if returncode_val == 0 else ("cancelled" if self._canceled else "failed"),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        # Add error context for failures
        if returncode_val != 0 and log_lines:
            meta_payload["error_logs"] = log_lines[-50:]
        
        # Check both global telemetry AND per-run metadata settings
        should_emit_metadata = self._telemetry_enabled and settings.get("save_metadata", True)
        if should_emit_metadata:
            try:
                # Emit for all runs (success, failure, cancellation)
                metadata_target = Path(output_path) if Path(output_path).exists() or Path(output_path).parent.exists() else self.output_dir
                if png_output:
                    write_png_metadata(metadata_target if metadata_target.is_dir() else metadata_target.parent, meta_payload)
                else:
                    emit_metadata(metadata_target, meta_payload)
            except Exception:
                pass  # Don't fail run if metadata fails
        
        # Log command to executed_commands folder
        try:
            self._command_logger.log_command(
                tab_name="rife",
                command=cmd,
                settings=settings,
                returncode=returncode_val,
                output_path=output_path,
                error_logs=log_lines[-50:] if returncode_val != 0 else None,
                execution_time=execution_time,
                additional_info={
                    "mode": "subprocess",
                    "cancelled": self._canceled,
                    "png_output": png_output,
                    "trimmed_input": str(trimmed_video) if trimmed_video else None
                }
            )
            if on_progress:
                on_progress("✅ Command logged to executed_commands folder\n")
        except Exception as e:
            if on_progress:
                on_progress(f"⚠️ Failed to log command: {e}\n")
        
        # FIXED: Clean up trimmed temp file if we created one
        if trimmed_video and trimmed_video.exists():
            try:
                trimmed_video.unlink()
                if on_progress:
                    on_progress("✅ Cleaned up temporary trimmed video\n")
            except Exception:
                pass  # Non-critical cleanup failure

        return RunResult(returncode_val, output_path, "\n".join(log_lines))

    def _build_rife_cmd(
        self,
        cli_path: Path,
        input_path: str,
        output_path: Path,
        settings: Dict[str, Any],
    ) -> List[str]:
        cmd: List[str] = [self._get_python_executable(), str(cli_path)]

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

        # Frame control: skip_first_frames and load_cap are now handled via ffmpeg preprocessing
        # in run_rife() before RIFE CLI is called, so no CLI args needed here.
        # RIFE's --skip flag is for static frame detection, not frame trimming.

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


