import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def _sanitize_name(name: str) -> str:
    """Make a filesystem-safe filename from a preset or model name."""
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(name))
    return safe.strip("._") or "default"


class PresetManager:
    """
    File-based preset manager.

    Layout:
    presets/
      global.json
      <tab>/
        <model>/
          last_used_preset.txt
          <preset_name>.json
    """

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Core helpers
    # ------------------------------------------------------------------ #
    def _tab_dir(self, tab: str) -> Path:
        return self.base_dir / _sanitize_name(tab)

    def _model_dir(self, tab: str, model: Optional[str]) -> Path:
        model_name = _sanitize_name(model) if model else "default"
        return self._tab_dir(tab) / model_name

    def _preset_path(self, tab: str, model: Optional[str], preset_name: str) -> Path:
        return self._model_dir(tab, model) / f"{_sanitize_name(preset_name)}.json"

    def _last_used_path(self, tab: str, model: Optional[str]) -> Path:
        return self._model_dir(tab, model) / "last_used_preset.txt"

    # ------------------------------------------------------------------ #
    # Global settings
    # ------------------------------------------------------------------ #
    def load_global_settings(self, defaults: Dict[str, Any]) -> Dict[str, Any]:
        path = self.base_dir / "global.json"
        if not path.exists():
            return defaults.copy()
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            merged = defaults.copy()
            merged.update(data)
            return merged
        except Exception:
            return defaults.copy()

    def save_global_settings(self, settings: Dict[str, Any]) -> None:
        path = self.base_dir / "global.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
        tmp_path.replace(path)

    # ------------------------------------------------------------------ #
    # Preset operations
    # ------------------------------------------------------------------ #
    def list_presets(self, tab: str, model: Optional[str]) -> List[str]:
        folder = self._model_dir(tab, model)
        if not folder.exists():
            return []
        return sorted([p.stem for p in folder.glob("*.json")])

    def save_preset(self, tab: str, model: Optional[str], preset_name: str, data: Dict[str, Any]) -> str:
        folder = self._model_dir(tab, model)
        folder.mkdir(parents=True, exist_ok=True)
        preset_path = self._preset_path(tab, model, preset_name)
        tmp_path = preset_path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp_path.replace(preset_path)
        # Track last used
        self.set_last_used(tab, model, preset_name)
        return preset_path.stem

    def load_preset(self, tab: str, model: Optional[str], preset_name: str) -> Optional[Dict[str, Any]]:
        preset_path = self._preset_path(tab, model, preset_name)
        if not preset_path.exists():
            return None
        try:
            with preset_path.open("r", encoding="utf-8") as f:
                preset_data = json.load(f)
            
            # Validate and clean loaded preset
            cleaned = self.validate_and_clean_preset(preset_data, tab, model)
            
            # Apply tab-specific constraints
            validated = self.validate_preset_constraints(cleaned, tab, model)
            
            # Tab-specific validation on load
            if tab == "seedvr2":
                # Validate batch_size 4n+1 formula
                batch_size = validated.get("batch_size", 5)
                if (batch_size - 1) % 4 != 0:
                    corrected = max(5, (batch_size // 4) * 4 + 1)
                    validated["batch_size"] = corrected
                    print(f"⚠️ Preset '{preset_name}' had invalid batch_size {batch_size}, auto-corrected to {corrected}")
                
                # Check cache + multi-GPU conflict
                cache_enabled = validated.get("cache_dit") or validated.get("cache_vae")
                cuda_device = str(validated.get("cuda_device", ""))
                if cache_enabled and cuda_device:
                    devices = [d.strip() for d in cuda_device.split(",") if d.strip()]
                    if len(devices) > 1:
                        validated["cache_dit"] = False
                        validated["cache_vae"] = False
                        print(f"⚠️ Preset '{preset_name}' had cache enabled with multi-GPU, auto-disabled caching")
            
            return validated
        except Exception as e:
            print(f"Error loading preset {preset_name} for {tab}/{model}: {e}")
            return None

    def delete_preset(self, tab: str, model: Optional[str], preset_name: str) -> bool:
        preset_path = self._preset_path(tab, model, preset_name)
        if preset_path.exists():
            preset_path.unlink()
            return True
        return False

    # ------------------------------------------------------------------ #
    # Last used tracking
    # ------------------------------------------------------------------ #
    def set_last_used(self, tab: str, model: Optional[str], preset_name: str) -> None:
        path = self._last_used_path(tab, model)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write(_sanitize_name(preset_name))

    def get_last_used_name(self, tab: str, model: Optional[str]) -> Optional[str]:
        path = self._last_used_path(tab, model)
        if not path.exists():
            return None
        try:
            return path.read_text(encoding="utf-8").strip() or None
        except Exception:
            return None

    def load_last_used(self, tab: str, model: Optional[str]) -> Optional[Dict[str, Any]]:
        name = self.get_last_used_name(tab, model)
        if not name:
            return None
        return self.load_preset(tab, model, name)

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #
    @staticmethod
    def merge_config(current: Dict[str, Any], preset: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge preset values onto current values, preserving current values for missing preset keys.
        
        This allows older presets to work with new features - if a preset doesn't have a key,
        the current default value is preserved.
        """
        if not preset:
            return current.copy()
        
        merged = current.copy()
        
        # Only update keys that exist in the preset
        # Keys in current that aren't in preset are preserved with their default values
        for key, value in preset.items():
            # Type checking to ensure compatibility
            if key in current:
                # Try to preserve type of current value
                try:
                    current_type = type(current[key])
                    if current_type in (int, float, str, bool):
                        # Convert to expected type
                        merged[key] = current_type(value)
                    else:
                        # For complex types, use as-is
                        merged[key] = value
                except (ValueError, TypeError):
                    # If conversion fails, use default
                    merged[key] = current[key]
            else:
                # New key from preset not in current defaults - add it
                merged[key] = value
        
        return merged

    @staticmethod
    def validate_and_clean_preset(preset: Dict[str, Any], tab: str, model: Optional[str] = None) -> Dict[str, Any]:
        """Validate and clean preset data before saving/loading."""
        if not isinstance(preset, dict):
            return {}

        cleaned = {}
        for key, value in preset.items():
            # Basic type validation - ensure values are JSON serializable
            try:
                import json
                json.dumps({key: value})  # Test serialization
                cleaned[key] = value
            except (TypeError, ValueError):
                # Skip non-serializable values with warning
                print(f"Warning: Skipping non-serializable preset value for key '{key}' in {tab}/{model}")
                continue

        return cleaned

    def save_preset_safe(self, tab: str, model: Optional[str], preset_name: str, data: Dict[str, Any]) -> str:
        """Safe preset saving with validation and error handling."""
        try:
            # Validate and clean the data
            cleaned_data = self.validate_and_clean_preset(data, tab, model)
            if not cleaned_data:
                raise ValueError("No valid data to save")

            # Save the preset
            result = self.save_preset(tab, model, preset_name, cleaned_data)
            return result
        except Exception as e:
            print(f"Error saving preset {preset_name} for {tab}/{model}: {e}")
            raise

    def load_preset_safe(self, tab: str, model: Optional[str], preset_name: str) -> Optional[Dict[str, Any]]:
        """Safe preset loading with error handling."""
        try:
            preset = self.load_preset(tab, model, preset_name)
            if preset is not None:
                # Validate loaded data
                cleaned = self.validate_and_clean_preset(preset, tab, model)
                return cleaned
            return None
        except Exception as e:
            print(f"Error loading preset {preset_name} for {tab}/{model}: {e}")
            return None

    def validate_preset_constraints(self, preset: Dict[str, Any], tab: str, model: Optional[str] = None) -> Dict[str, Any]:
        """Apply tab-specific validation constraints to presets."""
        validated = preset.copy()

        if tab == "seedvr2":
            # SeedVR2 specific validations
            validated = self._validate_seedvr2_constraints(validated)
        elif tab == "gan":
            # GAN specific validations
            validated = self._validate_gan_constraints(validated)
        elif tab == "flashvsr":
            # FlashVSR+ specific validations
            validated = self._validate_flashvsr_constraints(validated)
        elif tab == "rife":
            # RIFE specific validations
            validated = self._validate_rife_constraints(validated)

        return validated

    def _validate_seedvr2_constraints(self, preset: Dict[str, Any]) -> Dict[str, Any]:
        """Apply SeedVR2-specific validation rules."""
        validated = preset.copy()

        # Batch size must be 4n+1
        try:
            bs = int(validated.get("batch_size", 5))
            if bs % 4 != 1:
                validated["batch_size"] = max(1, (bs // 4) * 4 + 1)
        except (ValueError, TypeError):
            validated["batch_size"] = 5  # Safe default

        # Resolution must be multiple of 16
        try:
            res = int(validated.get("resolution", 1080))
            if res % 16 != 0:
                validated["resolution"] = (res // 16) * 16
        except (ValueError, TypeError):
            validated["resolution"] = 1080

        # Max resolution validation
        try:
            max_res = int(validated.get("max_resolution", 0))
            if max_res < 0:
                validated["max_resolution"] = 0
        except (ValueError, TypeError):
            validated["max_resolution"] = 0

        # VAE tiling constraints
        if validated.get("vae_encode_tiled"):
            try:
                tile_size = int(validated.get("vae_encode_tile_size", 1024))
                overlap = int(validated.get("vae_encode_tile_overlap", 128))
                if overlap >= tile_size:
                    validated["vae_encode_tile_overlap"] = max(0, tile_size - 1)
                if tile_size < 64:
                    validated["vae_encode_tile_size"] = 512
            except (ValueError, TypeError):
                validated["vae_encode_tile_size"] = 1024
                validated["vae_encode_tile_overlap"] = 128

        if validated.get("vae_decode_tiled"):
            try:
                tile_size = int(validated.get("vae_decode_tile_size", 1024))
                overlap = int(validated.get("vae_decode_tile_overlap", 128))
                if overlap >= tile_size:
                    validated["vae_decode_tile_overlap"] = max(0, tile_size - 1)
                if tile_size < 64:
                    validated["vae_decode_tile_size"] = 512
            except (ValueError, TypeError):
                validated["vae_decode_tile_size"] = 1024
                validated["vae_decode_tile_overlap"] = 128

        # BlockSwap requires dit_offload_device
        try:
            blocks_to_swap = int(validated.get("blocks_to_swap", 0))
            swap_io = bool(validated.get("swap_io_components", False))
            blockswap_enabled = blocks_to_swap > 0 or swap_io
            if blockswap_enabled:
                offload_device = str(validated.get("dit_offload_device", "none")).lower().strip()
                if offload_device in ("none", ""):
                    validated["dit_offload_device"] = "cpu"
        except (ValueError, TypeError):
            pass

        # Multi-GPU constraints - improved parsing
        try:
            cuda_device_str = str(validated.get("cuda_device", "0"))
            # Remove all whitespace and split by comma
            devices = [d.strip() for d in cuda_device_str.replace(" ", "").split(",") if d.strip() and d.strip().isdigit()]
            
            if len(devices) > 1:
                # Multi-GPU detected - disable cache options
                if validated.get("cache_dit"):
                    validated["cache_dit"] = False
                if validated.get("cache_vae"):
                    validated["cache_vae"] = False
        except Exception:
            # If parsing fails, assume single GPU
            pass

        # Validate compile cache limits
        try:
            cache_limit = int(validated.get("compile_dynamo_cache_size_limit", 64))
            if cache_limit < 1:
                validated["compile_dynamo_cache_size_limit"] = 64
        except (ValueError, TypeError):
            validated["compile_dynamo_cache_size_limit"] = 64

        try:
            recompile_limit = int(validated.get("compile_dynamo_recompile_limit", 128))
            if recompile_limit < 1:
                validated["compile_dynamo_recompile_limit"] = 128
        except (ValueError, TypeError):
            validated["compile_dynamo_recompile_limit"] = 128

        # Validate noise scales (0.0 to 1.0)
        for noise_key in ["input_noise_scale", "latent_noise_scale"]:
            try:
                noise_val = float(validated.get(noise_key, 0.0))
                validated[noise_key] = max(0.0, min(1.0, noise_val))
            except (ValueError, TypeError):
                validated[noise_key] = 0.0

        # Validate seed
        try:
            seed = int(validated.get("seed", 42))
            # Allow -1 for random
            if seed < -1:
                validated["seed"] = 42
        except (ValueError, TypeError):
            validated["seed"] = 42

        return validated

    def _validate_gan_constraints(self, preset: Dict[str, Any]) -> Dict[str, Any]:
        """Apply GAN-specific validation rules."""
        # GAN models typically don't have the same constraints as SeedVR2
        # Add any GAN-specific validations here if needed
        return preset
    
    def _validate_flashvsr_constraints(self, preset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply FlashVSR+-specific validation rules using metadata registry.
        
        Enforces:
        - Single GPU requirement (multi-GPU not supported)
        - Tile size/overlap constraints
        - Valid version/mode/scale combinations
        """
        validated = preset.copy()
        
        # Import metadata functions
        from shared.models.flashvsr_meta import get_flashvsr_metadata
        
        # Build model identifier
        model_id = f"v{validated.get('version', '10')}_{validated.get('mode', 'tiny')}_{validated.get('scale', 4)}x"
        model_meta = get_flashvsr_metadata(model_id)
        
        if model_meta:
            # Enforce single GPU
            device_str = str(validated.get("device", "auto"))
            if device_str not in ("auto", "cpu", ""):
                devices = [d.strip() for d in device_str.replace(" ", "").split(",") if d.strip()]
                if len(devices) > 1:
                    validated["device"] = devices[0]  # Use first GPU only
                    print(f"WARNING: FlashVSR+ preset: Multi-GPU not supported, using single GPU: {devices[0]}")
            
            # Validate tile constraints
            if validated.get("tiled_vae") or validated.get("tiled_dit"):
                tile_size = int(validated.get("tile_size", 256))
                overlap = int(validated.get("overlap", 24))
                
                if overlap >= tile_size:
                    validated["overlap"] = max(0, tile_size - 1)
                    print(f"WARNING: FlashVSR+ preset: Tile overlap >= tile size, correcting to {validated['overlap']}")
                
                if tile_size < 64:
                    validated["tile_size"] = model_meta.default_tile_size
                    print(f"WARNING: FlashVSR+ preset: Tile size too small, resetting to {validated['tile_size']}")
        
        return validated
    
    def _validate_rife_constraints(self, preset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply RIFE-specific validation rules using metadata registry.
        
        Enforces:
        - Single GPU requirement (multi-GPU not supported)
        - FPS multiplier limits from model metadata
        - Scale factor validation
        """
        validated = preset.copy()
        
        # Import metadata functions
        from shared.models.rife_meta import get_rife_metadata, get_rife_default_model
        
        # Get model metadata
        model_name = validated.get("rife_model", get_rife_default_model())
        model_meta = get_rife_metadata(model_name)
        
        if model_meta:
            # Enforce single GPU
            gpu_device_str = str(validated.get("gpu_device", ""))
            if gpu_device_str and gpu_device_str not in ("", "cpu"):
                devices = [d.strip() for d in gpu_device_str.replace(" ", "").split(",") if d.strip()]
                if len(devices) > 1:
                    validated["gpu_device"] = devices[0]  # Use first GPU only
                    print(f"WARNING: RIFE preset: Multi-GPU not supported, using single GPU: {devices[0]}")
            
            # Validate FPS multiplier
            fps_mult_str = str(validated.get("fps_multiplier", "x2"))
            try:
                mult_value = int(fps_mult_str.replace("x", "").strip())
                max_mult = model_meta.max_fps_multiplier
                
                if mult_value > max_mult:
                    validated["fps_multiplier"] = f"x{max_mult}"
                    print(f"WARNING: RIFE preset: FPS multiplier {mult_value}x exceeds limit {max_mult}x, clamping")
            except (ValueError, AttributeError):
                pass  # Keep original if parsing fails
            
            # Validate scale factor
            try:
                scale = float(validated.get("scale", 1.0))
                if scale <= 0:
                    validated["scale"] = 1.0
                    print(f"WARNING: RIFE preset: Invalid scale {scale}, resetting to 1.0")
            except (ValueError, TypeError):
                validated["scale"] = 1.0
        
        return validated


