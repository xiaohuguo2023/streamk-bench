import unittest
import os
from kernel_generator import KernelGenerator

class TestKernelGenerator(unittest.TestCase):

    def setUp(self):
        # Initializing with some example parameters
        self.kernel_gen = KernelGenerator(
            M=64, N=64, K=64,
            col_a=0, col_b=0,
            dtype_a='float32', dtype_b='float32', dtype_c='float32',
            init_type='random', configs=[{'block_m': 16, 'block_n': 16, 'block_k': 16, 'split_k': 1}],
            jobs=1, iters=1, run_bench=False,
            rotating_buffer_size=16, bias_size=0, icache_flush=False
        )
        self.file_paths = ['test_kernel_0.py']

    def tearDown(self):
        # Cleanup any files created during tests
        for file_path in self.file_paths:
            if os.path.exists(file_path):
                os.remove(file_path)

    def test_initialization(self):
        self.assertEqual(self.kernel_gen.M, 64)
        self.assertEqual(self.kernel_gen.N, 64)
        self.assertEqual(self.kernel_gen.K, 64)
        self.assertEqual(self.kernel_gen.dtype_a, 'float32')
        self.assertEqual(self.kernel_gen.dtype_b, 'float32')
        self.assertEqual(self.kernel_gen.dtype_c, 'float32')
        self.assertEqual(len(self.kernel_gen.configs), 1)
        self.assertFalse(self.kernel_gen.icache_flush)

    def test_write_imports(self):
        with open(self.file_paths[0], 'w') as f_kernel:
            self.kernel_gen.write_imports([f_kernel])
        
        with open(self.file_paths[0], 'r') as f_kernel:
            content = f_kernel.read()
        
        self.assertIn('import torch', content)
        self.assertIn('import triton', content)
        self.assertIn('import argparse', content)
        self.assertNotIn('from icache_flush import icache_flush', content)

    def test_write_kernel_definitions(self):
        # Mocking the matmul_kernel file content
        matmul_kernel_content = '''
        @triton.jit
        def matmul_kernel():
            pass
        '''
        with open(os.path.dirname(os.path.abspath(__file__)) + "/matmul_kernel.py", 'w') as f:
            f.write(matmul_kernel_content)

        with open(self.file_paths[0], 'w') as f_kernel:
            self.kernel_gen.write_kernel_definitions([f_kernel], self.kernel_gen.jobs, self.kernel_gen.configs, 
                                                     self.kernel_gen.M, self.kernel_gen.N, self.kernel_gen.K, 
                                                     self.kernel_gen.dtype_a, self.kernel_gen.dtype_b, 
                                                     self.kernel_gen.dtype_c, self.kernel_gen.bias_size)
          
        with open(self.file_paths[0], 'r') as f_kernel:
            content = f_kernel.read()
        
        self.assertIn('def matmul_kernel_', content)
        self.assertIn('def try_config_', content)
        self.assertIn('BLOCK_SIZE_M', content)

if __name__ == '__main__':
    unittest.main()
