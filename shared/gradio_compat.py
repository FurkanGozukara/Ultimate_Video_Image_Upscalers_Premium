"""
Gradio Compatibility Check and Feature Detection

Verifies installed Gradio version and feature availability to prevent runtime errors.
Provides graceful fallbacks for missing features.
"""

import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger("GradioCompat")


def check_gradio_version() -> Tuple[bool, str, Dict[str, Any]]:
    """
    Check installed Gradio version and feature availability.
    
    Returns:
        (is_compatible, message, features_dict)
    """
    try:
        import gradio as gr
        
        # Get version
        version = getattr(gr, '__version__', 'unknown')
        
        # Parse version for comparison
        try:
            major, minor, patch = map(int, version.split('.')[:3])
            version_tuple = (major, minor, patch)
        except Exception:
            version_tuple = (0, 0, 0)
            
        # Minimum required version: 4.0.0 (for ImageSlider, Timer, etc.)
        min_version = (4, 0, 0)
        
        is_compatible = version_tuple >= min_version
        
        # Feature detection
        features = {
            "version": version,
            "version_tuple": version_tuple,
            "ImageSlider": hasattr(gr, 'ImageSlider'),
            "Timer": hasattr(gr, 'Timer'),
            "themes": hasattr(gr, 'themes'),
            "HTML": hasattr(gr, 'HTML'),
            "Gallery": hasattr(gr, 'Gallery'),
            "Video": hasattr(gr, 'Video'),
            "File": hasattr(gr, 'File'),
            "State": hasattr(gr, 'State'),
            "Progress": hasattr(gr, 'Progress'),
        }
        
        # Build message
        if is_compatible:
            msg = f"✅ Gradio {version} is compatible (minimum: 4.0.0)"
            
            # Check for missing optional features
            missing = [k for k, v in features.items() if k not in ("version", "version_tuple") and not v]
            if missing:
                msg += f"\n⚠️ Missing optional features: {', '.join(missing)}"
        else:
            msg = f"❌ Gradio {version} is too old (minimum: 4.0.0)\n"
            msg += "Please upgrade: pip install --upgrade gradio"
            
        logger.info(f"Gradio compatibility check: {msg}")
        
        return is_compatible, msg, features
        
    except ImportError:
        msg = "❌ Gradio is not installed! Please install: pip install gradio"
        logger.error(msg)
        return False, msg, {}
    except Exception as e:
        msg = f"❌ Failed to check Gradio compatibility: {str(e)}"
        logger.error(msg)
        return False, msg, {}


def get_safe_component(component_name: str, fallback=None):
    """
    Safely get a Gradio component with fallback.
    
    Args:
        component_name: Name of the component (e.g., 'ImageSlider')
        fallback: Fallback component if not available
        
    Returns:
        Component class or fallback
    """
    try:
        import gradio as gr
        
        component = getattr(gr, component_name, None)
        if component is not None:
            return component
            
        logger.warning(f"Gradio component '{component_name}' not available, using fallback")
        return fallback
        
    except Exception as e:
        logger.error(f"Error getting Gradio component '{component_name}': {e}")
        return fallback


def check_required_features() -> Tuple[bool, str]:
    """
    Check if all required Gradio features are available for the app.
    
    Returns:
        (all_present, message)
    """
    try:
        import gradio as gr
        
        required_features = [
            'Blocks',
            'Tab',
            'Row',
            'Column',
            'Button',
            'Textbox',
            'Dropdown',
            'Checkbox',
            'Slider',
            'Number',
            'File',
            'Image',
            'Video',
            'Markdown',
            'HTML',
            'State',
        ]
        
        missing = []
        for feature in required_features:
            if not hasattr(gr, feature):
                missing.append(feature)
        
        if missing:
            msg = f"❌ Missing required Gradio features: {', '.join(missing)}\n"
            msg += "Please upgrade Gradio: pip install --upgrade gradio"
            return False, msg
        
        return True, "✅ All required Gradio features are available"
        
    except Exception as e:
        return False, f"❌ Failed to check Gradio features: {str(e)}"


def warn_deprecated_features():
    """
    Check for deprecated features that might be removed in future Gradio versions.
    
    Returns:
        List of warnings
    """
    warnings = []
    
    try:
        import gradio as gr
        
        # Check version for known deprecations
        version = getattr(gr, '__version__', 'unknown')
        
        try:
            major, minor, _ = map(int, version.split('.')[:3])
            
            # Gradio 4.x deprecation warnings
            if major >= 4:
                # Check for old-style component usage patterns
                if hasattr(gr, 'outputs'):
                    warnings.append("⚠️ gr.outputs is deprecated in Gradio 4.x. Use component classes directly.")
                
                if hasattr(gr, 'inputs'):
                    warnings.append("⚠️ gr.inputs is deprecated in Gradio 4.x. Use component classes directly.")
                    
        except Exception:
            pass
            
    except Exception as e:
        logger.error(f"Error checking deprecated features: {e}")
        
    return warnings


def get_compatibility_report() -> str:
    """
    Generate a comprehensive compatibility report.
    
    Returns:
        Multi-line compatibility report string
    """
    is_compatible, version_msg, features = check_gradio_version()
    all_required, features_msg = check_required_features()
    deprecation_warnings = warn_deprecated_features()
    
    report = []
    report.append("=== Gradio Compatibility Report ===\n")
    report.append(version_msg)
    report.append(features_msg)
    
    if features:
        report.append("\nFeature Availability:")
        for feature, available in features.items():
            if feature not in ("version", "version_tuple"):
                status = "✅" if available else "❌"
                report.append(f"  {status} {feature}")
    
    if deprecation_warnings:
        report.append("\nDeprecation Warnings:")
        for warning in deprecation_warnings:
            report.append(f"  {warning}")
    
    if is_compatible and all_required:
        report.append("\n✅ Gradio setup is fully compatible with this application")
    else:
        report.append("\n❌ Gradio setup has compatibility issues - please upgrade")
        
    return "\n".join(report)


# Auto-check on import
if __name__ != "__main__":
    is_compatible, msg, features = check_gradio_version()
    if not is_compatible:
        logger.warning(f"Gradio compatibility issue: {msg}")
