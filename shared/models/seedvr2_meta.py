from dataclasses import dataclass
from typing import Dict, List

from pathlib import Path

DEFAULT_BATCH_SIZE = 5
DEFAULT_ATTENTION = "sdpa"

# Local scan will look for common weight extensions inside ./models/seedvr2 (and siblings).
MODEL_EXTS = {".safetensors", ".gguf"}


@dataclass
class SeedVR2Model:
    name: str
    size: str
    precision: str
    variant: str
    supports_multi_gpu: bool = True
    default_batch_size: int = DEFAULT_BATCH_SIZE
    preferred_attention: str = DEFAULT_ATTENTION
    compile_compatible: bool = True
    max_resolution: int = 0  # 0 = no cap


def _built_ins() -> List[SeedVR2Model]:
    return [
        SeedVR2Model(name="seedvr2_ema_3b_fp8_e4m3fn.safetensors", size="3B", precision="fp8_e4m3fn", variant="standard", preferred_attention="flash_attn", supports_multi_gpu=True),
        SeedVR2Model(name="seedvr2_ema_3b_fp16.safetensors", size="3B", precision="fp16", variant="standard", preferred_attention="flash_attn", supports_multi_gpu=True),
        # GGUF models don't support multi-GPU well due to quantization overhead
        SeedVR2Model(name="seedvr2_ema_3b-Q8_0.gguf", size="3B", precision="Q8_0", variant="standard", compile_compatible=False, supports_multi_gpu=False),
        SeedVR2Model(name="seedvr2_ema_3b-Q4_K_M.gguf", size="3B", precision="Q4_K_M", variant="standard", compile_compatible=False, supports_multi_gpu=False),
        SeedVR2Model(name="seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors", size="7B", precision="fp8_e4m3fn_mixed_block35_fp16", variant="standard", preferred_attention="flash_attn", supports_multi_gpu=True),
        SeedVR2Model(name="seedvr2_ema_7b_fp16.safetensors", size="7B", precision="fp16", variant="standard", preferred_attention="flash_attn", supports_multi_gpu=True),
        SeedVR2Model(name="seedvr2_ema_7b-Q4_K_M.gguf", size="7B", precision="Q4_K_M", variant="standard", compile_compatible=False, supports_multi_gpu=False),
        SeedVR2Model(name="seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors", size="7B", precision="fp8_e4m3fn_mixed_block35_fp16", variant="sharp", preferred_attention="flash_attn", supports_multi_gpu=True),
        SeedVR2Model(name="seedvr2_ema_7b_sharp_fp16.safetensors", size="7B", precision="fp16", variant="sharp", preferred_attention="flash_attn", supports_multi_gpu=True),
        SeedVR2Model(name="seedvr2_ema_7b_sharp-Q4_K_M.gguf", size="7B", precision="Q4_K_M", variant="sharp", compile_compatible=False, supports_multi_gpu=False),
    ]


def _scan_local_weights() -> List[str]:
    """
    Discover locally downloaded SeedVR2 weights to keep dropdowns in sync with the filesystem.
    Searches ./models/seedvr2 relative to repository root (shared/models/../../models/seedvr2).
    """
    try:
        root = Path(__file__).resolve().parents[2]
        candidates = [
            root / "models" / "seedvr2",
            root / "models" / "SeedVR2",
        ]
        found: List[str] = []
        for base in candidates:
            if not base.exists():
                continue
            for f in base.iterdir():
                if f.is_file() and f.suffix.lower() in MODEL_EXTS:
                    found.append(f.name)
        return sorted(list({*found}))
    except Exception:
        return []


def get_seedvr2_models() -> List[SeedVR2Model]:
    disk = _scan_local_weights()
    built = _built_ins()
    # Add disk-only models with conservative defaults
    known_names = {m.name for m in built}
    extras = [
        SeedVR2Model(name=name, size="unknown", precision="unknown", variant="custom", supports_multi_gpu=True)
        for name in disk
        if name not in known_names
    ]
    return built + extras


def get_seedvr2_model_names() -> List[str]:
    return [m.name for m in get_seedvr2_models()]


def model_meta_map() -> Dict[str, SeedVR2Model]:
    return {m.name: m for m in get_seedvr2_models()}


