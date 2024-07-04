class ConfigSpaces:
    @staticmethod
    def get_full_tuning_space():
        configs = []

        block_mn_range = [16, 32, 64, 128, 256]
        block_k_range = [16, 32, 64, 128, 256]
        split_k_range = [1, 2, 4, 5, 6, 8, 10, 12, 16, 18, 24]
        num_warps_range = [1, 2, 4, 8]
        group_m_range = [1, 4, 8, 16, 32]
        # For now we see better perf with num_stages=0 for all gemm configs we care
        # But keep this explicit so that we do not forget we may need to set it to
        # other values in the future
        num_stage_range = [0]
        waves_per_eu_range = [0]
        matrix_instr_nonkdim_range = [16, 32]
        kpack_range = [1, 2]

        for block_m in block_mn_range:
            for block_n in block_mn_range:
                for block_k in block_k_range:
                    for num_warps in num_warps_range:
                        for group_m in group_m_range:
                            for split_k in split_k_range:
                                for num_stages in num_stage_range:
                                    for waves_per_eu in waves_per_eu_range:
                                        for matrix_instr_nonkdim in matrix_instr_nonkdim_range:
                                            for kpack in kpack_range:
                                                configs.append({'BLOCK_SIZE_M': block_m, 'BLOCK_SIZE_N': block_n, 'BLOCK_SIZE_K': block_k, 'GROUP_SIZE_M': group_m, 'SPLIT_K': split_k, 'num_warps': num_warps, 'num_stages': num_stages, 'waves_per_eu': waves_per_eu, 'matrix_instr_nonkdim': matrix_instr_nonkdim, 'kpack': kpack})

        return configs

    @staticmethod
    def get_default_config():                                                                                                       full_configs = get_full_tuning_space()                                                                                      return full_configs[0]

    @staticmethod
    def prune_configs(M, N, K, configs, elemBytes_a, elemBytes_b):
        pruned_configs = []

        if M < 32 or N < 32:
            mfma = 16
        else:
            mfma = 32

        # TODO (zhanglx): figure out the boundary between large and small gemms
        large_gemm = False
        if M >= 2048 and N >=2048:
            large_gemm = True

        for config in configs:
            BLOCK_SIZE_M = config.get("BLOCK_SIZE_M")
            BLOCK_SIZE_N = config.get("BLOCK_SIZE_N")
            BLOCK_SIZE_K = config.get("BLOCK_SIZE_K")
            num_warps = config.get("num_warps")
            num_stages = config.get("num_stages")
            matrix_instr_nonkdim = config.get("matrix_instr_nonkdim")
            kpack = config.get("kpack")
            if matrix_instr_nonkdim > mfma:
                continue
            if mfma == 4 and BLOCK_SIZE_K < 64:
                continue
            # some layouts could not work properly in case
            # number elemens per thread is less 1
            if BLOCK_SIZE_M * BLOCK_SIZE_N < 64:
                continue
            SPLIT_K = config.get("SPLIT_K")
            GROUP_M = config.get("GROUP_SIZE_M")
            if BLOCK_SIZE_M < matrix_instr_nonkdim or BLOCK_SIZE_N < matrix_instr_nonkdim:
                continue
            if M <= matrix_instr_nonkdim and BLOCK_SIZE_M != matrix_instr_nonkdim:
                continue
            if N <= matrix_instr_nonkdim and BLOCK_SIZE_N != matrix_instr_nonkdim:
                continue
            # Skip BLOCK_SIZE that is too large compare to M/N
            # unless BLOCK_SIZE is already small enough
            if BLOCK_SIZE_M > M * 2 and BLOCK_SIZE_M != 16:
                continue
            if BLOCK_SIZE_N > N * 2 and BLOCK_SIZE_N != 16:
                continue
            # skip large split_k when not necessary
            if SPLIT_K != 1 and not need_split_k(M, N, K):
                continue
            # skip split_k that leads to EVEN_K = false
            leap = SPLIT_K * BLOCK_SIZE_K
            modv = K % leap
            if modv != 0:
                continue
            # skip large GROUP_M
            if GROUP_M * BLOCK_SIZE_M > M and GROUP_M != 1:
                continue
            # out of shared memory resource
            # TODO (zhanglx): This does not consider the LDS usage in the epilogue
            LDS = BLOCK_SIZE_K * BLOCK_SIZE_M * elemBytes_a + BLOCK_SIZE_K * BLOCK_SIZE_N * elemBytes_b
            LDS = LDS if not num_stages else LDS * num_stages
            if LDS > 65536:
                continue
            # Skip small block sizes and num_warps for large gemm
            # For fp16 and f8, we want to only use BLOCK_SIZE >= 64
            if large_gemm:
                if BLOCK_SIZE_M < 64 or BLOCK_SIZE_N < 64:
                    continue
                if BLOCK_SIZE_K < 64:
                    continue
                if num_warps < 4:
                    continue
                # check if tiling is integer multiple of GEMM size because we have no boundary check
                if M % BLOCK_SIZE_M != 0 or N % BLOCK_SIZE_N != 0 or K % BLOCK_SIZE_K != 0:
                    continue

            pruned_configs.append(config)

        return pruned_configs

    @staticmethod
    def need_split_k(SIZE_M, SIZE_N, SIZE_K):
        return (SIZE_M < 64 or SIZE_N < 64) and SIZE_K > 1024

    @staticmethod
    def read_config(config):
        block_m = config.get('BLOCK_SIZE_M')
        block_n = config.get('BLOCK_SIZE_N')
        block_k = config.get('BLOCK_SIZE_K')
        group_m = config.get('GROUP_SIZE_M')
        split_k = config.get('SPLIT_K')
        num_warps = config.get('num_warps')
        num_stages = config.get('num_stages')
        waves_per_eu = config.get('waves_per_eu')
        mfma_instr_size = config.get('matrix_instr_nonkdim')
        kpack = config.get('kpack')
        return block_m, block_n, block_k, group_m, split_k, num_warps, num_stages, waves_per_eu, mfma_instr_size, kpack

