from pathlib import Path
from typing import Dict, Any
import gradio as gr

from shared.health import collect_health_report


def build_health_callbacks(global_settings: Dict[str, Any], health_banner: Dict[str, str]):
    def health_check_action():
        report = collect_health_report(temp_dir=Path(global_settings["temp_dir"]), output_dir=Path(global_settings["output_dir"]))
        lines = []
        warnings = []
        for key, info in report.items():
            line = f"**{key}**: {info.get('status')} - {info.get('detail')}"
            lines.append(line)
            if info.get("status") not in ("ok", "skipped"):
                warnings.append(line)
        health_banner["text"] = "\n".join(warnings) if warnings else "All health checks passed."
        return "\n".join(lines), health_banner["text"]

    return {"health_check_action": health_check_action}


