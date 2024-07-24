import math
from utils import Utils

def gen_rotating_tensors(M, N, K, dtype_a, need_Trans_a, dtype_b, need_Trans_b, dtype_c, seed, init_type, rotating_buffer_size, bias_size, device='cuda'):
    a_size = M * K * Utils.type_name_to_bytes(dtype_a)
    b_size = K * N * Utils.type_name_to_bytes(dtype_b)
    c_size = M * N * Utils.type_name_to_bytes(dtype_c)
    bias_size_bytes = bias_size * Utils.type_name_to_bytes(dtype_c)

    total_size = a_size + b_size + c_size + bias_size_bytes
    block_count = rotating_buffer_size * 1024 * 1024 // total_size
    block_count = max(1, block_count)

    # Generate input and outputs
    a = []
    b = []
    c = []
    bias = []
    for i in range(block_count):
        in_a, in_a_fp16 = InputGenerator(M, K, dtype_a, need_Trans_a, seed + i, init_type, device).generate()
        a.append(in_a)
        in_b, in_b_fp16 = InputGenerator(K, N, dtype_b, need_Trans_b, seed + i, init_type, device).generate()
        b.append(in_b)
        out_c = torch.zeros((M, N), dtype=TypeMappings.tl_to_torch_types[TypeMappings.name_to_tl_types[dtype_c]], device=device)
        c.append(out_c)
        if bias_size > 0:
            bs, bs_fp16 = InputGenerator(M, 1, dtype_b, need_Trans_b, seed + i, init_type, device).generate()
            bias.append(bs.squeeze())

    in_outs = {
        "rotating_num": block_count,
        "input_a": a,
        "input_b": b,
        "output_c": c,
        "bias": bias
    }

    return in_outs
