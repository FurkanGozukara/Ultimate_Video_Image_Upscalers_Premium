import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


class RunLogger:
    """Persist per-run JSON summaries alongside outputs."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def write_summary(self, output_path: Path, payload: Dict[str, Any]) -> Optional[Path]:
        if not self.enabled:
            return None
        target_dir = output_path if output_path.is_dir() else output_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        summary_path = target_dir / "run_summary.json"
        payload = payload.copy()
        payload.setdefault("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return summary_path


