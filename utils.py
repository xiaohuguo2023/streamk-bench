class Utils:
    @staticmethod
    def get_default_tuning_result_filename():
        git_branch_name = run_bash_command("git rev-parse --abbrev-ref HEAD")
        git_branch_name = git_branch_name[0].decode()
        # handle branch name of "xxx/xxx" format
        git_branch_name = git_branch_name.replace('/', '_')
        git_commit_hash = run_bash_command("git rev-parse --short HEAD")
        git_commit_hash = git_commit_hash[0].decode()

        dt_string = datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
        defaultName = f"tuning_results_{git_branch_name}@{git_commit_hash}_{dt_string}.yaml"
        return defaultName


    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(
            prog="tune a specific gemm size",
            allow_abbrev=False,
        )

        parser.add_argument("-m", type=int, default=0)
        parser.add_argument("-n", type=int, default=0)
        parser.add_argument("-k", type=int, default=0)
        parser.add_argument("-col_a", action='store_true', default=False, help='whether matrix a is column major')
        parser.add_argument("-col_b", action='store_true', default=False, help='whether matrix b is column major')
        parser.add_argument("-dtype_a", type=str, default='fp16', help="matrix a element data type")
        parser.add_argument("-dtype_b", type=str, default='fp16', help="matrix b element data type")
        parser.add_argument("-dtype_c", type=str, default='fp16', help="output element data type")
        parser.add_argument("--ngpus", type=int, default=0, help='number of GPUs used in the profiling step')
        parser.add_argument("--gpu_ids", type=lambda s: [int(id) for id in s.split(',')], default=[], help='list of gpu ids to use for tuning')
        parser.add_argument("--gemm_size_file", type=str, default="", help='yaml file to indicate matrix size')
        parser.add_argument("--o", type=str, default='', help='yaml file to store tuning results')
        parser.add_argument("--keep", action='store_true', default=False, help='keep generated files')
        parser.add_argument("--compare", action='store_true', default=False, help="Whether check result correctness")
        parser.add_argument("--compare_wo_tuning", action='store_true', default=False, help="Whether check result correctness")
        parser.add_argument("--benchmark", action='store_true', default=False, help="Benchmark the given config")
        parser.add_argument("--time_breakdown", action='store_true', default=False, help="Show detailed time breakdown of each step during the tuning")
        parser.add_argument("--verbose", action='store_true', default=False, help="enables time_breakdown and additional logging messages")
        parser.add_argument("--num_threads", type=int, default=16, help="number of threads to use for kernel compilation and post processing")
        parser.add_argument("--jobs", type=int, default=1, help="number of generated files")
        parser.add_argument("--iters", type=int, default=1000, help="number of generated files")
        parser.add_argument("--init_type", type=str, default='randn', help="Initialization type for input matrices (default uniform rand [0, 1.0)])")
        parser.add_argument("--rotating_tensor", type=int, default=0, help="total size (MB) of all tensors (default 0 MB (no rotating tensor), need to be larger than the L1, L2, MALL size)")
        parser.add_argument("--bias_vector", action='store_true', default=False, help="apply bias vector")
        parser.add_argument("--icache_flush", action='store_true', default=False, help="apply icache flush in tuning performance")
        parser.add_argument("--no_warmup", action='store_true', default=False, help="Do not call the warmup kernel")
        args = parser.parse_args()
        if not args.o:
            if args.benchmark:
                args.o = "benchmarking_results.csv"
            else:
                args.o = get_default_tuning_result_filename()

        return args

    @staticmethod
    def process_item(item):
        M = item['M']
        N = item['N']
        K = item['K']
        col_a = False if item['rowMajorA'] == 'T' else True
        col_b = False if item['rowMajorB'] == 'T' else True
        del item['M']
        del item['N']
        del item['K']
        del item['rowMajorA']
        del item['rowMajorB']
        return M, N, K, col_a, col_b, item

    @staticmethod
    def type_name_to_bytes(ty_name):
        type_bytes = {
            'float32': 4,
            'fp32': 4,
            'float16': 2,
            'fp16': 2,
            'float8e4b8': 1,
            'float8e5b16': 1,
        }
        if ty_name in type_bytes:
            return type_bytes[ty_name]
        else:
            raise ValueError(f"Unrecognized input type name {ty_name}")
            sys.exit(1)

    @staticmethod
    def format_output(unformatted):
        if unformatted < 0.0001:
            formatted = "{:.3e}".format(unformatted)
        elif unformatted > 1000:
            formatted = "{:.1f}".format(unformatted)
        else:
            formatted = "{:.2f}".format(unformatted)
        return formatted

    @staticmethod
    def extract_kernel_time(M, N, K, config, df, bias_size):
        configStr, _ = gen_kernel_and_configStr_from_config(M, N, K, config, None, None, None, bias_size)
        df = df[df['KernelName'].str.contains(configStr)]
        meanTime = df['DurationNs'].tail(100).mean()
        return config, meanTime

    @staticmethod
    def profile_batch_kernels(M, N, K, gpuid, gpus, jobs, verbose):
        ngpus = len(gpus)
        gpuIdx = gpus.index(gpuid)
        if gpuIdx + 1 > jobs:
            return
        os.environ['ROCR_VISIBLE_DEVICES'] = str(gpuid)
        jobId = gpuIdx
        while jobId < jobs:
            kernel_name = generated_kernel_name(M, N, K, jobId)
            if verbose:
                print(f"profiling {kernel_name} on GPU {gpuid}")
            run_bash_command_wrapper(f"rocprof --stats -o results-{jobId}.csv python {kernel_name}", capture=(verbose < 2))
            jobId += ngpus

    #Todo, need have a common wrapper for different gemm kernel
    @staticmethod
    def matmul(a, b, c, bias, block_m, block_n, block_k, group_m, split_k, num_warps, num_stages, waves_per_eu, mfmaInstrSize, kpack, use_bias):
        # Check constraints.
        assert a.shape[1] == b.shape[0], "Incompatible dimensions"
        #assert a.is_contiguous(), "Matrix A must be contiguous"
        #assert b.is_contiguous(), "Matrix B must be contiguous"
        M, K = a.shape
        K, N = b.shape
        # 1D launch kernel where each block gets its own program.

        grid = triton.cdiv(M, block_m) * triton.cdiv(N, block_n), split_k
        stride_bias = bias.stride(0) if use_bias else 0
        matmul_kernel[grid](
            a, b, c, bias,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            stride_bias=stride_bias,
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
            GROUP_SIZE_M=group_m,
            SPLIT_K=split_k,
            num_warps=num_warps,
            num_stages=num_stages,
            waves_per_eu=waves_per_eu,
            matrix_instr_nonkdim=mfmaInstrSize,
            kpack=kpack,
            BIAS=use_bias,
        )
        return c
