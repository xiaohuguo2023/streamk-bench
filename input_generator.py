import torch
import triton
import triton.language as tl
from type_mappings import TypeMappings
from utils import Utils

class InputGenerator:
    def __init__(self, M, N, ty_name, needTrans, seed, init_type, device='cuda'):
        self.M = M
        self.N = N
        self.ty_name = ty_name
        self.needTrans = needTrans
        self.seed = seed
        self.init_type = init_type
        self.device = device
        self.d_type = TypeMappings.name_to_tl_types[ty_name]
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    @staticmethod
    @triton.jit
    def copy_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        input = tl.load(input_ptr + offsets, mask=mask)
        output = input
        tl.store(output_ptr + offsets, output, mask=mask)

    def init_by_size_and_type(self, size, dtype, init_type):
        if init_type == 'hpl':
            return torch.empty(size, device='cuda', dtype=dtype).uniform_(-0.5, 0.5)
        elif init_type == 'trig_float':
            M, N = size
            return torch.reshape(torch.arange(0, M*N), (M, N)).sin().to(dtype=dtype, device='cuda')
        elif init_type == 'zeros':
            return torch.zeros(size, dtype=dtype, device='cuda')
        elif init_type == "randn":
            return torch.randn(size, dtype=dtype, device='cuda')
        else:
            raise ValueError("Bad matrix initialization type.")

    def generate(self):
        raw_data = self.init_by_size_and_type((self.N, self.M) if self.needTrans else (self.M, self.N), torch.float32, self.init_type)
        if self.needTrans:
            raw_data = raw_data.T
        
        if (self.d_type == tl.float8e4b8 and TypeMappings.TORCH_HAS_FP8E4B8) or \
           (self.d_type == tl.float8e5b16 and TypeMappings.TORCH_HAS_FP8E5B16) or not self.d_type.is_fp8():
            input = raw_data.to(TypeMappings.tl_to_torch_types[self.d_type])
            input_f16 = input.to(torch.float16)
        else:
            f8_tensor = raw_data.to(torch.int8)
            # keep only two bits of exponent to avoid overflow
            f8_tensor = f8_tensor & 0b00111111
            input = triton.reinterpret(f8_tensor, self.d_type)
            input_f16 = torch.empty_like(f8_tensor, dtype=torch.float16)
            n_elements = raw_data.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            self.copy_kernel[grid](input, input_f16, n_elements, BLOCK_SIZE=1024)

        return input, input_f16

    @staticmethod
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
