from dataclasses import dataclass
from typing import Dict, List

from pathlib import Path

DEFAULT_BATCH_SIZE = 5
DEFAULT_ATTENTION = "sdpa"


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
        SeedVR2Model(name="seedvr2_ema_3b_fp8_e4m3fn.safetensors", size="3B", precision="fp8_e4m3fn", variant="standard"),
        SeedVR2Model(name="seedvr2_ema_3b_fp16.safetensors", size="3B", precision="fp16", variant="standard"),
        SeedVR2Model(name="seedvr2_ema_3b-Q8_0.gguf", size="3B", precision="Q8_0", variant="standard"),
        SeedVR2Model(name="seedvr2_ema_3b-Q4_K_M.gguf", size="3B", precision="Q4_K_M", variant="standard"),
        SeedVR2Model(name="seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors", size="7B", precision="fp8_e4m3fn_mixed_block35_fp16", variant="standard"),
        SeedVR2Model(name="seedvr2_ema_7b_fp16.safetensors", size="7B", precision="fp16", variant="standard"),
        SeedVR2Model(name="seedvr2_ema_7b-Q4_K_M.gguf", size="7B", precision="Q4_K_M", variant="standard"),
        SeedVR2Model(name="seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors", size="7B", precision="fp8_e4m3fn_mixed_block35_fp16", variant="sharp"),
        SeedVR2Model(name="seedvr2_ema_7b_sharp_fp16.safetensors", size="7B", precision="fp16", variant="sharp"),
        SeedVR2Model(name="seedvr2_ema_7b_sharp-Q4_K_M.gguf", size="7B", precision="Q4_K_M", variant="sharp"),
    ]


def get_seedvr2_models() -> List[SeedVR2Model]:
    return _built_ins()


def get_seedvr2_model_names() -> List[str]:
    return [m.name for m in get_seedvr2_models()]


def model_meta_map() -> Dict[str, SeedVR2Model]:
    return {m.name: m for m in get_seedvr2_models()}


