from typing import Any, Dict

import gradio as gr

from ui.shared_components import preset_section


def build_face_tab(
    defaults: Dict[str, Any],
    preset_manager,
    global_settings: Dict[str, Any],
    callbacks: Dict[str, Any],
):
    """
    Face Restoration tab UI builder.
    callbacks must provide:
      - order list
      - models()
      - refresh_presets(model, select_name=None)
      - save_preset(name, *args)
      - load_preset(name, model, defaults, current_values)
      - safe_defaults()
      - set_face_global(val)
    """
    values = [defaults[k] for k in callbacks["order"]]

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### Face Restoration")
            model_select = gr.Dropdown(label="Model", choices=callbacks["models"](), value=values[0])
            enable_face = gr.Checkbox(label="Enable Face Restoration", value=values[1])
            strength = gr.Slider(label="Strength", minimum=0.0, maximum=1.0, step=0.05, value=values[2])
            apply_globally = gr.Checkbox(label="Apply globally to all runs", value=values[3])
            info = gr.Markdown("Global toggle applies to single and batch runs. Settings are saved per model.")
        with gr.Column(scale=2):
            preset_dropdown, preset_name, save_preset_btn, load_preset_btn, preset_status, safe_defaults_btn = preset_section(
                "Face",
                preset_manager,
                values[0],
                preset_manager.list_presets("face", values[0]),
                preset_manager.get_last_used_name("face", values[0]),
                safe_defaults_label="Safe Defaults (Face)",
            )

    inputs_list = [model_select, enable_face, strength, apply_globally]

    save_preset_btn.click(
        fn=callbacks["save_preset"],
        inputs=[preset_name] + inputs_list,
        outputs=[preset_dropdown, preset_status] + inputs_list,
    )
    load_preset_btn.click(
        fn=lambda preset, model, *vals: callbacks["load_preset"](preset, model, defaults, list(vals)),
        inputs=[preset_dropdown, model_select] + inputs_list,
        outputs=inputs_list,
    )
    safe_defaults_btn.click(fn=callbacks["safe_defaults"], outputs=inputs_list)

    # Hook global apply toggle to global settings
    apply_globally.change(fn=callbacks["set_face_global"], inputs=[apply_globally], outputs=info)
    strength.change(fn=callbacks["cache_strength"], inputs=[strength], outputs=info)

