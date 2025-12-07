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
        """Merge preset values onto current values without overwriting missing keys."""
        if not preset:
            return current
        merged = current.copy()
        for key, value in preset.items():
            merged[key] = value
        return merged


