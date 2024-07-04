import torch
import triton.language as tl

class TypeMappings:
    TORCH_HAS_FP8E5B16 = hasattr(torch, 'float8_e5m2fnuz')
    TORCH_HAS_FP8E4B8 = hasattr(torch, 'float8_e4m3fnuz')

    tl_to_torch_types = {
        tl.float16: torch.float16,
        tl.bfloat16: torch.bfloat16,
        tl.float32: torch.float32,
        tl.int8: torch.int8,
        tl.int32: torch.int32,
    }

    if TORCH_HAS_FP8E5B16:
        tl_to_torch_types[tl.float8e5b16] = torch.float8_e5m2fnuz

    if TORCH_HAS_FP8E4B8:
        tl_to_torch_types[tl.float8e4b8] = torch.float8_e4m3fnuz

    name_to_tl_types = {
        'int8': tl.int8,
        'int32': tl.int32,
        'fp16': tl.float16,
        'fp32': tl.float32,
        'bf16': tl.bfloat16,
        'fp8': tl.float8e4b8,
        'bf8': tl.float8e5b16,
    }
