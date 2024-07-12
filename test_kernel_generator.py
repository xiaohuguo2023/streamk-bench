import unittest
import os
import subprocess
from kernel_generator import KernelGenerator

class TestKernelGenerator(unittest.TestCase):

    def setUp(self):
        self.kernel_gen = KernelGenerator(
            M=64, N=64, K=64,
            col_a=0, col_b=0,
            dtype_a='fp32', dtype_b='fp32', dtype_c='fp32',
            init_type='randn', configs=[{'block_m': 16, 'block_n': 16, 'block_k': 16, 'split_k': 1}],
            jobs=24, iters=1, run_bench=False,
            rotating_buffer_size=16, bias_size=0, icache_flush=False
        )

    def tearDown(self):
        for i in range(24):
            filename = self.kernel_gen.generated_kernel_name(64, 64, 64, i)
            if os.path.exists(filename):
                os.remove(filename)

    def test_generate_and_run_kernel(self):
        self.kernel_gen.generate_kernel(
            M=64, N=64, K=64,
            col_a=0, col_b=0,
            dtype_a='fp32', dtype_b='fp32', dtype_c='fp32',
            init_type='randn', configs=[{'block_m': 16, 'block_n': 16, 'block_k': 16, 'split_k': 1}],
            jobs=24, iters=1, run_bench=False,
            rotating_buffer_size=16, bias_size=0, icache_flush=False
        )

        for i in range(24):
            filename = self.kernel_gen.generated_kernel_name(64, 64, 64, i)
            self.assertTrue(os.path.exists(filename), f"Kernel file {filename} was not created.")
            
            # Run the generated kernel script to verify its correctness
            result = subprocess.run(['python', filename], capture_output=True, text=True)
            if result.returncode != 0:
                # If there is an error, print the contents of the generated kernel file for debugging
                with open(filename, 'r') as file:
                    kernel_content = file.read()
                    print(f"Contents of {filename}:\n{kernel_content}")

            self.assertEqual(result.returncode, 0, f"Kernel script {filename} did not run successfully. Output: {result.stdout} {result.stderr}")

if __name__ == '__main__':
    unittest.main()
