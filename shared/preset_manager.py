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
                return json.load(f)
        except Exception:
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
        """Merge preset values onto current values, preserving current values for missing preset keys."""
        if not preset:
            return current
        merged = current.copy()
        for key, value in preset.items():
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

        return validated

    def _validate_seedvr2_constraints(self, preset: Dict[str, Any]) -> Dict[str, Any]:
        """Apply SeedVR2-specific validation rules."""
        validated = preset.copy()

        # Batch size must be 4n+1
        bs = int(validated.get("batch_size", 5))
        if bs % 4 != 1:
            validated["batch_size"] = max(1, (bs // 4) * 4 + 1)

        # VAE tiling constraints
        if validated.get("vae_encode_tiled"):
            tile_size = validated.get("vae_encode_tile_size", 1024)
            overlap = validated.get("vae_encode_tile_overlap", 128)
            if overlap >= tile_size:
                validated["vae_encode_tile_overlap"] = max(0, tile_size - 1)

        if validated.get("vae_decode_tiled"):
            tile_size = validated.get("vae_decode_tile_size", 1024)
            overlap = validated.get("vae_decode_tile_overlap", 128)
            if overlap >= tile_size:
                validated["vae_decode_tile_overlap"] = max(0, tile_size - 1)

        # BlockSwap requires dit_offload_device
        blockswap_enabled = validated.get("blocks_to_swap", 0) > 0 or validated.get("swap_io_components", False)
        if blockswap_enabled and str(validated.get("dit_offload_device", "none")).lower() in ("none", ""):
            validated["dit_offload_device"] = "cpu"

        # Multi-GPU constraints
        devices = [d.strip() for d in str(validated.get("cuda_device", "")).split(",") if d.strip()]
        if len(devices) > 1:
            if validated.get("cache_dit"):
                validated["cache_dit"] = False
            if validated.get("cache_vae"):
                validated["cache_vae"] = False

        return validated

    def _validate_gan_constraints(self, preset: Dict[str, Any]) -> Dict[str, Any]:
        """Apply GAN-specific validation rules."""
        # GAN models typically don't have the same constraints as SeedVR2
        # Add any GAN-specific validations here if needed
        return preset


