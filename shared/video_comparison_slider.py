"""
Custom HTML5 Video Comparison Slider Component

Since Gradio's ImageSlider doesn't support videos, this provides
a custom HTML/JavaScript implementation for side-by-side video comparison
with an interactive slider.

Features:
- Side-by-side video comparison with vertical slider
- Synchronized playback
- Fullscreen support
- Play/pause controls
- Timeline scrubbing
- Responsive design
"""

from pathlib import Path
from typing import Optional, Tuple


def create_video_comparison_html(
    original_video: Optional[str],
    upscaled_video: Optional[str],
    height: int = 600,
    slider_position: float = 50.0
) -> str:
    """
    Create HTML for video comparison slider.
    
    Args:
        original_video: Path to original video file
        upscaled_video: Path to upscaled video file
        height: Height of video player in pixels
        slider_position: Initial slider position (0-100%)
        
    Returns:
        HTML string with embedded JavaScript for video comparison
    """
    
    if not original_video or not upscaled_video:
        return """
        <div style="text-align: center; padding: 40px; background: #f0f0f0; border-radius: 8px;">
            <p style="color: #666; font-size: 16px;">Upload and upscale videos to see comparison</p>
        </div>
        """
    
    # Generate HTML with embedded JavaScript
    html = f"""
    <div id="video-comparison-container" style="position: relative; width: 100%; max-width: 1200px; margin: 0 auto; background: #000; border-radius: 8px; overflow: hidden;">
        <!-- Video container -->
        <div id="video-wrapper" style="position: relative; width: 100%; height: {height}px; overflow: hidden;">
            <!-- Original video (left side) -->
            <div id="original-side" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; overflow: hidden;">
                <video id="original-video" style="width: 100%; height: 100%; object-fit: contain;" preload="metadata">
                    <source src="file={original_video}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </div>
            
            <!-- Upscaled video (right side) -->
            <div id="upscaled-side" style="position: absolute; top: 0; right: 0; width: 50%; height: 100%; overflow: hidden;">
                <video id="upscaled-video" style="width: 200%; height: 100%; object-fit: contain; margin-left: -100%;" preload="metadata">
                    <source src="file={upscaled_video}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </div>
            
            <!-- Slider handle -->
            <div id="slider-handle" style="position: absolute; top: 0; bottom: 0; width: 4px; background: #fff; box-shadow: 0 0 10px rgba(0,0,0,0.5); cursor: ew-resize; z-index: 10; left: 50%;">
                <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 40px; height: 40px; background: #fff; border-radius: 50%; box-shadow: 0 2px 8px rgba(0,0,0,0.3); display: flex; align-items: center; justify-content: center;">
                    <span style="color: #333; font-weight: bold;">⟷</span>
                </div>
            </div>
            
            <!-- Labels -->
            <div style="position: absolute; top: 10px; left: 10px; background: rgba(0,0,0,0.7); color: #fff; padding: 5px 10px; border-radius: 4px; font-size: 14px; z-index: 5;">
                Original
            </div>
            <div style="position: absolute; top: 10px; right: 10px; background: rgba(0,0,0,0.7); color: #fff; padding: 5px 10px; border-radius: 4px; font-size: 14px; z-index: 5;">
                Upscaled
            </div>
        </div>
        
        <!-- Controls -->
        <div id="controls" style="padding: 15px; background: #1a1a1a; display: flex; gap: 15px; align-items: center; flex-wrap: wrap;">
            <button id="play-pause-btn" style="padding: 8px 16px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; min-width: 80px;">
                ▶️ Play
            </button>
            
            <div style="flex: 1; min-width: 200px;">
                <input id="timeline" type="range" min="0" max="100" value="0" style="width: 100%; cursor: pointer;">
            </div>
            
            <span id="time-display" style="color: #fff; font-family: monospace; font-size: 14px; min-width: 100px; text-align: right;">
                0:00 / 0:00
            </span>
            
            <button id="fullscreen-btn" style="padding: 8px 16px; background: #555; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 14px;">
                ⛶ Fullscreen
            </button>
            
            <button id="sync-btn" style="padding: 8px 16px; background: #2196F3; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 14px;">
                🔄 Re-sync
            </button>
        </div>
    </div>
    
    <script>
    (function() {{
        const container = document.getElementById('video-comparison-container');
        const originalVideo = document.getElementById('original-video');
        const upscaledVideo = document.getElementById('upscaled-video');
        const sliderHandle = document.getElementById('slider-handle');
        const upscaledSide = document.getElementById('upscaled-side');
        const playPauseBtn = document.getElementById('play-pause-btn');
        const timeline = document.getElementById('timeline');
        const timeDisplay = document.getElementById('time-display');
        const fullscreenBtn = document.getElementById('fullscreen-btn');
        const syncBtn = document.getElementById('sync-btn');
        const videoWrapper = document.getElementById('video-wrapper');
        
        let isDragging = false;
        let sliderPosition = {slider_position};
        
        // Initialize slider position
        updateSliderPosition(sliderPosition);
        
        // Slider dragging
        sliderHandle.addEventListener('mousedown', (e) => {{
            isDragging = true;
            e.preventDefault();
        }});
        
        document.addEventListener('mousemove', (e) => {{
            if (isDragging) {{
                const rect = videoWrapper.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const percentage = (x / rect.width) * 100;
                sliderPosition = Math.max(0, Math.min(100, percentage));
                updateSliderPosition(sliderPosition);
            }}
        }});
        
        document.addEventListener('mouseup', () => {{
            isDragging = false;
        }});
        
        // Touch support for mobile
        sliderHandle.addEventListener('touchstart', (e) => {{
            isDragging = true;
            e.preventDefault();
        }});
        
        document.addEventListener('touchmove', (e) => {{
            if (isDragging) {{
                const touch = e.touches[0];
                const rect = videoWrapper.getBoundingClientRect();
                const x = touch.clientX - rect.left;
                const percentage = (x / rect.width) * 100;
                sliderPosition = Math.max(0, Math.min(100, percentage));
                updateSliderPosition(sliderPosition);
            }}
        }});
        
        document.addEventListener('touchend', () => {{
            isDragging = false;
        }});
        
        function updateSliderPosition(percentage) {{
            sliderHandle.style.left = percentage + '%';
            upscaledSide.style.width = (100 - percentage) + '%';
        }}
        
        // Play/Pause
        playPauseBtn.addEventListener('click', () => {{
            if (originalVideo.paused) {{
                originalVideo.play();
                upscaledVideo.play();
                playPauseBtn.textContent = '⏸️ Pause';
            }} else {{
                originalVideo.pause();
                upscaledVideo.pause();
                playPauseBtn.textContent = '▶️ Play';
            }}
        }});
        
        // Timeline
        timeline.addEventListener('input', (e) => {{
            const time = (e.target.value / 100) * originalVideo.duration;
            originalVideo.currentTime = time;
            upscaledVideo.currentTime = time;
        }});
        
        // Update timeline and time display
        originalVideo.addEventListener('timeupdate', () => {{
            const percentage = (originalVideo.currentTime / originalVideo.duration) * 100;
            timeline.value = percentage || 0;
            
            const current = formatTime(originalVideo.currentTime);
            const total = formatTime(originalVideo.duration);
            timeDisplay.textContent = `${{current}} / ${{total}}`;
        }});
        
        // Sync button
        syncBtn.addEventListener('click', () => {{
            upscaledVideo.currentTime = originalVideo.currentTime;
            if (!originalVideo.paused) {{
                upscaledVideo.play();
            }} else {{
                upscaledVideo.pause();
            }}
        }});
        
        // Fullscreen
        fullscreenBtn.addEventListener('click', () => {{
            if (container.requestFullscreen) {{
                container.requestFullscreen();
            }} else if (container.webkitRequestFullscreen) {{
                container.webkitRequestFullscreen();
            }} else if (container.msRequestFullscreen) {{
                container.msRequestFullscreen();
            }}
        }});
        
        // Format time helper
        function formatTime(seconds) {{
            if (isNaN(seconds)) return '0:00';
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return `${{mins}}:${{secs.toString().padStart(2, '0')}}`;
        }}
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {{
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            
            switch(e.key) {{
                case ' ':
                    e.preventDefault();
                    playPauseBtn.click();
                    break;
                case 'f':
                case 'F':
                    fullscreenBtn.click();
                    break;
                case 'ArrowLeft':
                    originalVideo.currentTime = Math.max(0, originalVideo.currentTime - 5);
                    upscaledVideo.currentTime = originalVideo.currentTime;
                    break;
                case 'ArrowRight':
                    originalVideo.currentTime = Math.min(originalVideo.duration, originalVideo.currentTime + 5);
                    upscaledVideo.currentTime = originalVideo.currentTime;
                    break;
            }}
        }});
        
        // Auto-sync when videos end
        originalVideo.addEventListener('ended', () => {{
            playPauseBtn.textContent = '▶️ Play';
        }});
        
        // Load both videos
        originalVideo.load();
        upscaledVideo.load();
    }})();
    </script>
    
    <style>
    #video-comparison-container {{
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    }}
    
    #play-pause-btn:hover, #fullscreen-btn:hover, #sync-btn:hover {{
        opacity: 0.9;
        transform: translateY(-1px);
    }}
    
    #play-pause-btn:active, #fullscreen-btn:active, #sync-btn:active {{
        transform: translateY(0);
    }}
    
    #timeline {{
        accent-color: #4CAF50;
    }}
    
    #slider-handle:hover {{
        width: 6px;
    }}
    
    /* Fullscreen styles */
    #video-comparison-container:fullscreen {{
        display: flex;
        flex-direction: column;
    }}
    
    #video-comparison-container:fullscreen #video-wrapper {{
        flex: 1;
        height: auto !important;
    }}
    </style>
    """
    
    return html


def create_comparison_selector(original: Optional[str], upscaled: Optional[str]) -> str:
    """
    Create a video comparison component with native Gradio components where possible,
    falling back to custom HTML for video comparison.
    
    This is a wrapper that chooses the appropriate comparison method based on file types.
    """
    if not original or not upscaled:
        return create_video_comparison_html(None, None)
    
    # Check file types
    original_path = Path(original) if isinstance(original, str) else original
    upscaled_path = Path(upscaled) if isinstance(upscaled, str) else upscaled
    
    # If both are videos, use custom HTML slider
    video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    if (original_path.suffix.lower() in video_exts and 
        upscaled_path.suffix.lower() in video_exts):
        return create_video_comparison_html(str(original), str(upscaled))
    
    # For images, Gradio's ImageSlider can be used (handled in UI layer)
    # This is just a fallback message
    return """
    <div style="text-align: center; padding: 20px;">
        <p>Use ImageSlider component for image comparison</p>
    </div>
    """

