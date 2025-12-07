import gradio as gr


def build_health_tab(health_banner, callbacks):
    """
    Health Check tab UI builder.
    callbacks must provide:
      - run_health()
    """
    with gr.Tab("Health Check"):
        health_btn = gr.Button("Run Health Check")
        health_report = gr.Markdown("Run to verify ffmpeg, CUDA, VS Build Tools, and disk/temp/output writability.")
        health_btn.click(fn=callbacks["run_health"], outputs=health_report)
        return health_report

