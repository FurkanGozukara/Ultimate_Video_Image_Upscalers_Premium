"""
HTML comparison widgets for video and image with slider, swap, pin, and fullscreen.
Includes a lightweight fallback using the bundled Video_Comparison_Slider assets.
"""


def _pin_js(val: bool) -> str:
    return "true" if val else "false"


def build_video_comparison(
    input_video: str,
    output_video: str,
    pin_reference: bool = False,
    start_fullscreen: bool = False,
    use_fallback_assets: bool = False,
) -> str:
    if not input_video or not output_video:
        return "<p>No comparison available.</p>"
    if use_fallback_assets:
        try:
            import base64
            from pathlib import Path

            # Serve the bundled slider HTML (assumes Start_Video_Comparison assets are present)
            slider_html = Path(__file__).parent.parent / "Video_Comparison_Slider" / "index.html"
            if slider_html.exists():
                with slider_html.open("r", encoding="utf-8") as f:
                    html = f.read()
                html = html.replace("INPUT_PLACEHOLDER", f"file:///{input_video}")
                html = html.replace("OUTPUT_PLACEHOLDER", f"file:///{output_video}")
                return html
        except Exception:
            pass

    pin_js = _pin_js(pin_reference)
    fs_js = _pin_js(start_fullscreen)
    return f"""
<style>
.vid-cmp-shell {{
  width: 100%;
  max-width: 1080px;
  margin: 0 auto;
  font-family: Inter, system-ui, sans-serif;
}}
.vid-cmp-container {{
  position: relative;
  width: 100%;
  aspect-ratio: 16 / 9;
  background: #0f0f0f;
  overflow: hidden;
  border-radius: 10px;
  border: 1px solid #333;
}}
.vid-base, .vid-overlay {{
  width: 100%;
  height: 100%;
  object-fit: contain;
}}
.vid-overlay {{
  position: absolute;
  top: 0;
  left: 0;
  clip-path: inset(0 50% 0 0);
  pointer-events: none; /* Keep controls on base video */
}}
.vid-range {{
  width: 100%;
  margin-top: 10px;
  accent-color: #08f;
}}
.vid-toolbar {{
  display: flex;
  gap: 8px;
  margin-top: 8px;
  flex-wrap: wrap;
}}
.vid-toolbar button {{
  flex: 1 1 auto;
  min-width: 120px;
  padding: 8px 10px;
  background: #1e1e1e;
  color: #eee;
  border: 1px solid #333;
  border-radius: 6px;
  cursor: pointer;
}}
.vid-toolbar button:hover {{
  border-color: #08f;
}}
</style>
<div class="vid-cmp-shell">
  <div class="vid-cmp-container" id="vidCmpRoot">
    <video class="vid-base" id="vidBase" src="file:///{input_video}" controls></video>
    <video id="vidOverlay" class="vid-overlay" src="file:///{output_video}" controls muted></video>
  </div>
  <input id="vidRange" type="range" min="0" max="100" value="50" class="vid-range">
  <div class="vid-toolbar">
    <button id="swapBtn">Swap</button>
    <button id="pinBtn">Pin Reference</button>
    <button id="fsBtn">Fullscreen</button>
    <button id="resetBtn">Reset Slider</button>
  </div>
</div>
<script>
(() => {{
  const r = document.getElementById('vidRange');
  const o = document.getElementById('vidOverlay');
  const b = document.getElementById('vidBase');
  const swapBtn = document.getElementById('swapBtn');
  const pinBtn = document.getElementById('pinBtn');
  const fsBtn = document.getElementById('fsBtn');
  const resetBtn = document.getElementById('resetBtn');
  const root = document.getElementById('vidCmpRoot');

  let pinned = {pin_js};

  const applyClip = (val) => {{
    o.style.clipPath = `inset(0 ${100 - val}% 0 0)`;
  }};

  r.oninput = () => applyClip(r.value);
  applyClip(r.value);

  swapBtn.onclick = () => {{
    const tmp = o.src;
    o.src = b.src;
    b.src = tmp;
  }};

  pinBtn.onclick = () => {{
    pinned = !pinned;
    pinBtn.textContent = pinned ? "Unpin" : "Pin Reference";
    if (pinned) {{
      o.pause(); b.pause();
    }}
  }};

  fsBtn.onclick = () => {{
    if (root.requestFullscreen) root.requestFullscreen();
  }};

  resetBtn.onclick = () => {{
    r.value = 50;
    applyClip(50);
  }};

  // Initialize pin state
  if (pinned) {{
    pinBtn.textContent = "Unpin";
    o.pause(); b.pause();
  }}

  // Optional fullscreen on load
  if ({fs_js} && root.requestFullscreen) {{
    setTimeout(() => fsBtn.click(), 300);
  }}
}})();
</script>
"""


def build_image_comparison(input_path: str, output_path: str, pin_reference: bool = False) -> str:
    if not input_path or not output_path:
        return "<p>No comparison available.</p>"
    pin_js = _pin_js(pin_reference)
    return f"""
<style>
.img-cmp-shell {{
  width: 100%;
  max-width: 1080px;
  margin: 0 auto;
  font-family: Inter, system-ui, sans-serif;
}}
.img-cmp-container {{
  position: relative;
  width: 100%;
  overflow: hidden;
  border-radius: 10px;
  border: 1px solid #333;
}}
.img-base, .img-overlay {{
  width: 100%;
  display: block;
}}
.img-overlay {{
  position: absolute;
  top: 0;
  left: 0;
  clip-path: inset(0 50% 0 0);
}}
.img-range {{
  width: 100%;
  margin-top: 10px;
  accent-color: #08f;
}}
.img-toolbar {{
  display: flex;
  gap: 8px;
  margin-top: 8px;
  flex-wrap: wrap;
}}
.img-toolbar button {{
  flex: 1 1 auto;
  min-width: 120px;
  padding: 8px 10px;
  background: #1e1e1e;
  color: #eee;
  border: 1px solid #333;
  border-radius: 6px;
  cursor: pointer;
}}
.img-toolbar button:hover {{
  border-color: #08f;
}}
</style>
<div class="img-cmp-shell">
  <div class="img-cmp-container" id="imgCmpRoot">
    <img class="img-base" id="imgBase" src="file:///{input_path}" />
    <img class="img-overlay" id="imgOverlay" src="file:///{output_path}" />
  </div>
  <input id="imgRange" type="range" min="0" max="100" value="50" class="img-range">
  <div class="img-toolbar">
    <button id="imgSwapBtn">Swap</button>
    <button id="imgPinBtn">Pin Reference</button>
    <button id="imgResetBtn">Reset Slider</button>
  </div>
</div>
<script>
(() => {{
  const r = document.getElementById('imgRange');
  const o = document.getElementById('imgOverlay');
  const b = document.getElementById('imgBase');
  const swapBtn = document.getElementById('imgSwapBtn');
  const pinBtn = document.getElementById('imgPinBtn');
  const resetBtn = document.getElementById('imgResetBtn');

  let pinned = {pin_js};

  const applyClip = (val) => {{
    o.style.clipPath = `inset(0 ${100 - val}% 0 0)`;
  }};

  r.oninput = () => applyClip(r.value);
  applyClip(r.value);

  swapBtn.onclick = () => {{
    const tmp = o.src;
    o.src = b.src;
    b.src = tmp;
  }};

  pinBtn.onclick = () => {{
    pinned = !pinned;
    pinBtn.textContent = pinned ? "Unpin" : "Pin Reference";
  }};

  resetBtn.onclick = () => {{
    r.value = 50;
    applyClip(50);
  }};

  if (pinned) {{
    pinBtn.textContent = "Unpin";
  }}
}})();
</script>
"""

