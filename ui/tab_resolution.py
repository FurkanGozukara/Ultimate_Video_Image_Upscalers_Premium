from typing import Any, Dict

import gradio as gr

from ui.shared_components import preset_section


def build_resolution_tab(
    defaults: Dict[str, Any],
    preset_manager,
    seed_controls_cache: Dict[str, Any],
    callbacks: Dict[str, Any],
):
    """
    Resolution & Scene Split tab UI builder.
    callbacks must provide:
      - order: list of keys
      - refresh_presets(model, select_name=None)
      - save_preset(name, *args)
      - load_preset(name, model, defaults, current_values)
      - safe_defaults()
      - apply_to_seed(target_resolution, max_target_resolution)
      - chunk_estimate(size, overlap)
      - estimate_from_input(size, overlap)
      - cache_resolution(t_res, max_res)
      - cache_resolution_flags(auto_res, enable_max, chunk_size, chunk_overlap, ratio_down, per_cleanup)
    """
    values = [defaults[k] for k in callbacks["order"]]

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### Resolution & Scene Split")
            model_select = gr.Dropdown(label="Model", choices=callbacks["models"](), value=values[0])
            auto_resolution = gr.Checkbox(label="🎯 Auto-Resolution (Aspect Aware)", value=values[1])
            enable_max = gr.Checkbox(label="Enable Max Target Resolution", value=values[2])
            target_resolution = gr.Slider(label="Target Resolution", minimum=256, maximum=4096, step=16, value=values[3])
            max_target_resolution = gr.Slider(label="Max Target Resolution", minimum=0, maximum=8192, step=16, value=values[4])
            chunk_size = gr.Number(label="Chunk Size (0 = disabled)", value=values[5], precision=0)
            chunk_overlap = gr.Number(label="Chunk Overlap", value=values[6], precision=0)
            ratio_downscale = gr.Checkbox(label="Ratio Downscale-then-Upscale", value=values[7])
            per_chunk_cleanup = gr.Checkbox(label="Per-chunk temp cleanup", value=values[8])

            estimate_box = gr.Markdown("Chunk and resolution estimates will be shown here.")
            flags_msg = gr.Markdown("")

            apply_to_seed = gr.Button("Apply to SeedVR2")
        with gr.Column(scale=2):
            preset_dropdown, preset_name, save_preset_btn, load_preset_btn, preset_status, safe_defaults_btn = preset_section(
                "Resolution",
                preset_manager,
                values[0],
                preset_manager.list_presets("resolution", values[0]),
                preset_manager.get_last_used_name("resolution", values[0]),
                safe_defaults_label="Safe Defaults (Resolution)",
            )

    inputs_list = [
        model_select,
        auto_resolution,
        enable_max,
        target_resolution,
        max_target_resolution,
        chunk_size,
        chunk_overlap,
        ratio_downscale,
        per_chunk_cleanup,
    ]

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

    apply_to_seed.click(
        fn=callbacks["apply_to_seed"],
        inputs=[target_resolution, max_target_resolution],
        outputs=[callbacks["seed_controls"]["resolution"], callbacks["seed_controls"]["max_resolution"]],
    )

    chunk_size.change(
        fn=callbacks["chunk_estimate"],
        inputs=[chunk_size, chunk_overlap],
        outputs=estimate_box,
    )
    chunk_overlap.change(
        fn=callbacks["chunk_estimate"],
        inputs=[chunk_size, chunk_overlap],
        outputs=estimate_box,
    )

    estimate_from_input_box = gr.Markdown("")
    chunk_size.change(fn=callbacks["estimate_from_input"], inputs=[chunk_size, chunk_overlap], outputs=estimate_from_input_box)
    chunk_overlap.change(fn=callbacks["estimate_from_input"], inputs=[chunk_size, chunk_overlap], outputs=estimate_from_input_box)

    # cache for use in SeedVR2 run
    cache_msg = gr.Markdown("")
    target_resolution.change(
        fn=callbacks["cache_resolution"],
        inputs=[target_resolution, max_target_resolution, model_select],
        outputs=cache_msg,
    )
    max_target_resolution.change(
        fn=callbacks["cache_resolution"],
        inputs=[target_resolution, max_target_resolution, model_select],
        outputs=cache_msg,
    )
    auto_resolution.change(
        fn=callbacks["cache_resolution_flags"],
        inputs=[auto_resolution, enable_max, chunk_size, chunk_overlap, ratio_downscale, per_chunk_cleanup, model_select],
        outputs=flags_msg,
    )
    enable_max.change(
        fn=callbacks["cache_resolution_flags"],
        inputs=[auto_resolution, enable_max, chunk_size, chunk_overlap, ratio_downscale, per_chunk_cleanup, model_select],
        outputs=flags_msg,
    )
    chunk_size.change(
        fn=callbacks["cache_resolution_flags"],
        inputs=[auto_resolution, enable_max, chunk_size, chunk_overlap, ratio_downscale, per_chunk_cleanup, model_select],
        outputs=flags_msg,
    )
    chunk_overlap.change(
        fn=callbacks["cache_resolution_flags"],
        inputs=[auto_resolution, enable_max, chunk_size, chunk_overlap, ratio_downscale, per_chunk_cleanup, model_select],
        outputs=flags_msg,
    )
    ratio_downscale.change(
        fn=callbacks["cache_resolution_flags"],
        inputs=[auto_resolution, enable_max, chunk_size, chunk_overlap, ratio_downscale, per_chunk_cleanup, model_select],
        outputs=flags_msg,
    )
    per_chunk_cleanup.change(
        fn=callbacks["cache_resolution_flags"],
        inputs=[auto_resolution, enable_max, chunk_size, chunk_overlap, ratio_downscale, per_chunk_cleanup, model_select],
        outputs=flags_msg,
    )

    return {
        "target_resolution": target_resolution,
        "max_target_resolution": max_target_resolution,
    }

