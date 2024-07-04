import os

class KernelGenerator:

    def __init__(self, M, N, K, col_a, col_b, dtype_a, dtype_b, dtype_c, init_type, configs, jobs, iters, run_bench, rotating_buffer_size, bias_size, icache_flush):
        self.M = M
        self.N = N
        self.K = K
        self.col_a = col_a
        self.col_b = col_b
        self.dtype_a = dtype_a
        self.dtype_b = dtype_b
        self.dtype_c = dtype_c
        self.init_type = init_type
        self.configs = configs
        self.jobs = jobs
        self.iters = iters
        self.run_bench = run_bench
        self.rotating_buffer_size = rotating_buffer_size
        self.bias_size = bias_size
        self.icache_flush = icache_flush

    def write_imports(self, f_kernel):
        # write imports
        import_str = """import torch
    import triton
    import triton.language as tl
    import argparse
    import sys
    import multiprocessing
    from tune_gemm import gen_rotating_tensors
    """
        if icache_flush:
            import_str += """
    from icache_flush import icache_flush
    """
        for fi in range(jobs):
            f_kernel[fi].write(import_str + "\n")

    def write_kernel_definitions(f_kernel, jobs, configs, M, N, K, dtype_a, dtype_b, dtype_c, bias_size):
        # write definitions of matmul_kernel_xxx
        # and matmul_xxx and try_config
        with open(os.path.dirname(os.path.abspath(__file__))+"/matmul_kernel.py") as file:
            matmul_kernel_code = file.read()
        idx = 0
        for config in configs:
            file_idx = idx % jobs
            configStr, matmul_def_str = gen_kernel_and_configStr_from_config(M, N, K, config, dtype_a, dtype_b, dtype_c, bias_size)
            # Copy the matmul_kernel with name replaced
            matmul_kernel_config = matmul_kernel_code.replace("matmul_kernel", f"matmul_kernel_{configStr}")
            matmul_kernel_config = matmul_kernel_config.replace("import triton.language as tl", "")
            matmul_kernel_config = matmul_kernel_config.replace("import triton", "")
            f_kernel[file_idx].write(matmul_kernel_config + "\n\n")
            f_kernel[file_idx].write(matmul_def_str + "\n")
            idx += 1

    def write_test_gemm_preamble(f_kernel, jobs, M, N, K, dtype_a, col_a, dtype_b, col_b, dtype_c, init_type, rotating_buffer_size, bias_size):
        # write test_gemm
        # pre string
        test_gemm_pre_str = f"""def test_gemm(M, N, K, rotating_buffer_size, bias_size, num_threads):
        thread_pool = multiprocessing.Pool(processes=num_threads)
        tensors = gen_rotating_tensors(M, N, K, '{dtype_a}', {col_a}, '{dtype_b}', {col_b}, '{dtype_c}',
                                       1, '{init_type}', rotating_buffer_size, bias_size, device='cuda')

        a = tensors['input_a'][0]
        b = tensors['input_b'][0]
        c = tensors['output_c'][0]
        assert bias_size == M or bias_size == 0

        stride_bias = tensors['bias'][0].stride(0) if bias_size > 0 else 0
        task_args = (M, N, K,
                     a.stride(0), a.stride(1),
                     b.stride(0), b.stride(1),
                     c.stride(0), c.stride(1), stride_bias)

        if num_threads > 1:
            results = []
            config_names = []
    """
        for fi in range(jobs):
            f_kernel[fi].write(test_gemm_pre_str + "\n")

    def write_failed_config_handling(f_kernel, jobs, filenames):
        for fi in range(jobs):
            threadpool_str = """
            failed_configs = []
            for i in range(len(results)):
                results[i].wait()
                res = results[i].get()
                if not res:
                    failed_configs += [config_names[i]]
            thread_pool.close()
            thread_pool.join()
            with open("{filename}.failed_configs", "w") as f:
                for cfg in failed_configs:
                    f.write(cfg + "\\n")
        else:
            try:
                with open("{filename}.failed_configs", "r") as f:
                    failed_configs = [cfg.strip() for cfg in f.readlines()]
            except Exception:
                failed_configs = []
            """.format(filename=filenames[fi])
            f_kernel[fi].write(threadpool_str)

    def write_warm_up_calls(f_kernel, jobs, configs, M, N, K, bias_size):
        # warm up call of all matmul functions in parallel
        idx = 0
        for config in configs:
            configStr, _ = gen_kernel_and_configStr_from_config(M, N, K, config, None, None, None, bias_size)
            task_str = f"        results += [thread_pool.apply_async(try_config_{configStr}, args=task_args)]\n" + \
                       f"        config_names += ['{configStr}']\n"
            f_kernel[idx % jobs].write(task_str)
            idx += 1

    def write_gemm_function_calls(f_kernel, jobs, configs, M, N, K, iters, run_bench, icache_flush, bias_size):
        idx = 0
        runs = iters if run_bench else 200
        call_icache_flush = 'icache_flush()' if icache_flush else ''
        for config in configs:
            configStr, _ = gen_kernel_and_configStr_from_config(M, N, K, config, None, None, None, bias_size)
            matmul_call_str = f"""
            if '{configStr}' not in failed_configs:
                rotating_num = tensors['rotating_num']
                for i in range({runs}):
                    a = tensors['input_a'][i % rotating_num]
                    b = tensors['input_b'][i % rotating_num]
                    c = tensors['output_c'][i % rotating_num]
                    bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None"""
            if icache_flush:
                matmul_call_str += f"""
                    icache_flush()"""
            matmul_call_str += f"""
                    d = matmul_{configStr}(a, b, c, bias, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias.stride(0))"""
            f_kernel[idx % jobs].write(matmul_call_str + "\n")
            idx += 1
        # post string
        for fi in range(jobs):
            f_kernel[fi].write("        return d\n")

    def write_main_function(f_kernel, jobs, M, N, K, rotating_buffer_size):
        # def main and call test_gemm
        def_main_str = f"""
    def main():
        parser = argparse.ArgumentParser(
            prog="tune a specific gemm size",
            allow_abbrev=False,)
        parser.add_argument("-n", type=int, default=1, help='number of threads')
        parser.add_argument("-rotating_tensor", type=int, default={rotating_buffer_size}, help='size of rotating buffer (MB), default: 256')
        args = parser.parse_args()
        numThreads = args.n
        rotating_buffer_size = args.rotating_tensor
        """
        test_gemm_call_str = f'test_gemm({M}, {N}, {K}, rotating_buffer_size, {M}, numThreads)'
        for fi in range(jobs):
            f_kernel[fi].write(def_main_str)
            f_kernel[fi].write(test_gemm_call_str + "\n\n")
            f_kernel[fi].write("""if __name__ == '__main__':
       sys.exit(main())""")
            f_kernel[fi].close()

    def generate_kernel(M, N, K, col_a, col_b, dtype_a, dtype_b, dtype_c, init_type, configs, jobs, iters, run_bench, rotating_buffer_size, bias_size, icache_flush):

        filenames = [generated_kernel_name(M, N, K, i) for i in range(jobs)]
        f_kernel = [open(path, 'w') for path in filenames]

        write_imports(f_kernel, jobs, icache_flush)
        write_kernel_definitions(f_kernel, jobs, configs, M, N, K, dtype_a, dtype_b, dtype_c, bias_size)
        write_test_gemm_preamble(f_kernel, jobs, M, N, K, dtype_a, col_a, dtype_b, col_b, dtype_c, init_type, rotating_buffer_size, bias_size)
        write_warm_up_calls(f_kernel, jobs, configs, M, N, K, bias_size)
        write_failed_config_handling(f_kernel, jobs, filenames)
        write_gemm_function_calls(f_kernel, jobs, configs, M, N, K, iters, run_bench, icache_flush, bias_size)
        write_main_function(f_kernel, jobs, M, N, K, rotating_buffer_size)


    def generate_kernel_and_config_str(self, config, bias_size=None):
        # Implement your logic to generate kernel and config string based on the config and bias_size
        kernel_code = "generated_kernel_code"  # Replace with actual kernel generation logic
        config_str = "config_string"  # Replace with actual config string generation logic
        return kernel_code, config_str
