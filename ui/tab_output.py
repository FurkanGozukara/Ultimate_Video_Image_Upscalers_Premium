from typing import Any, Dict

import gradio as gr

from ui.shared_components import comparison_help, preset_section


def build_output_tab(
    defaults: Dict[str, Any],
    preset_manager,
    seed_controls_cache: Dict[str, Any],
    callbacks: Dict[str, Any],
):
    """
    Output & Comparison tab UI builder.
    callbacks must provide:
      - order list
      - refresh_presets(model, select_name=None)
      - save_preset(name, *args)
      - load_preset(name, model, defaults, current_values)
      - safe_defaults()
    """
    values = [defaults[k] for k in callbacks["order"]]

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### Output & Comparison")
            model_select = gr.Dropdown(label="Model", choices=callbacks["models"](), value=values[0])
            output_format = gr.Dropdown(label="Output Format", choices=["auto", "mp4", "png"], value=values[1])
            fps_override = gr.Number(label="FPS Override (0 = keep original)", value=values[2], precision=0)
            comparison_mode = gr.Dropdown(
                label="Comparison Mode",
                choices=["native", "html_slider", "fallback"],
                value=values[3],
                info="Fallback uses bundled slider assets.",
            )
            pin_reference = gr.Checkbox(label="Pin Reference", value=values[4])
            fullscreen = gr.Checkbox(label="Enable Fullscreen/Lightbox", value=values[5])
            apply_to_seed = gr.Button("Apply Output Format to SeedVR2")
            png_padding = gr.Number(label="PNG frame number padding", value=values[6], precision=0)
            png_keep_basename = gr.Checkbox(label="Keep input base-name for PNG sequences", value=values[7])
            skip_first_frames = gr.Number(label="Skip first frames (global hint)", value=values[8], precision=0)
            load_cap = gr.Number(label="Frame cap (global hint, 0 = all)", value=values[9], precision=0)

            comparison_note = comparison_help()
        with gr.Column(scale=2):
            preset_dropdown, preset_name, save_preset_btn, load_preset_btn, preset_status, safe_defaults_btn = preset_section(
                "Output",
                preset_manager,
                values[0],
                preset_manager.list_presets("output", values[0]),
                preset_manager.get_last_used_name("output", values[0]),
                safe_defaults_label="Safe Defaults (Output)",
            )

    inputs_list = [
        model_select,
        output_format,
        fps_override,
        comparison_mode,
        pin_reference,
        fullscreen,
        png_padding,
        png_keep_basename,
        skip_first_frames,
        load_cap,
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
        fn=lambda fmt: gr.Dropdown.update(value=fmt),
        inputs=[output_format],
        outputs=[callbacks["seed_controls"]["output_format"]],
    )

    def cache_output(fmt):
        seed_controls_cache["output_format_val"] = fmt
        return gr.Markdown.update(value="Output format cached for runs.")

    def cache_fps(fps_val):
        seed_controls_cache["fps_override_val"] = fps_val
        return gr.Markdown.update(value="FPS override cached for runs.")

    def cache_comparison(mode):
        seed_controls_cache["comparison_mode_val"] = mode
        return gr.Markdown.update(value="Comparison mode cached for runs.")

    def cache_pin(val):
        seed_controls_cache["pin_reference_val"] = bool(val)
        return gr.Markdown.update(value="Pin reference preference cached.")

    def cache_fullscreen(val):
        seed_controls_cache["fullscreen_val"] = bool(val)
        return gr.Markdown.update(value="Fullscreen preference cached.")

    def cache_png_padding(val):
        return callbacks["cache_png_padding"](val)

    def cache_png_basename(val):
        return callbacks["cache_png_basename"](val)

    def cache_skip(val):
        return callbacks["cache_skip"](val)

    def cache_cap(val):
        return callbacks["cache_cap"](val)

    cmp_cache_msg = gr.Markdown("")
    pin_cache_msg = gr.Markdown("")
    fs_cache_msg = gr.Markdown("")
    output_cache_msg = gr.Markdown("")
    fps_cache_msg = gr.Markdown("")
    png_pad_cache_msg = gr.Markdown("")
    png_base_cache_msg = gr.Markdown("")
    skip_cache_msg = gr.Markdown("")
    cap_cache_msg = gr.Markdown("")

    output_format.change(fn=cache_output, inputs=[output_format], outputs=output_cache_msg)
    fps_override.change(fn=cache_fps, inputs=[fps_override], outputs=fps_cache_msg)
    comparison_mode.change(fn=cache_comparison, inputs=[comparison_mode], outputs=cmp_cache_msg)
    pin_reference.change(fn=cache_pin, inputs=[pin_reference], outputs=pin_cache_msg)
    fullscreen.change(fn=cache_fullscreen, inputs=[fullscreen], outputs=fs_cache_msg)
    png_padding.change(fn=cache_png_padding, inputs=[png_padding], outputs=png_pad_cache_msg)
    png_keep_basename.change(fn=cache_png_basename, inputs=[png_keep_basename], outputs=png_base_cache_msg)
    skip_first_frames.change(fn=cache_skip, inputs=[skip_first_frames], outputs=skip_cache_msg)
    load_cap.change(fn=cache_cap, inputs=[load_cap], outputs=cap_cache_msg)


def comparison_note_block():
    return comparison_help()

