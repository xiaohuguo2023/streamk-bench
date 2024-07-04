import os
from subprocess

class SystemUtils:
    @staticmethod
    def is_hip_available():
				try:
						__import__("hip")
				except ImportError:
						return False
				else:
						return True

    @staticmethod
    def run_bash_command_wrapper(commandstring, capture=True):
        try:
            return SystemUtils.run_bash_command(commandstring, capture)
        except subprocess.CalledProcessError as e:
            if not capture:
                print(f"running {commandstring} one more time")
            return SystemUtils.run_bash_command(commandstring, capture)

    @staticmethod
    def run_bash_command(commandstring, capture=True):
				if capture:
						proc = subprocess.run(commandstring, shell=True, check=True, executable='/bin/bash', stdout=subprocess.PIPE)
						return proc.stdout.splitlines(), , proc.stderr.decode('utf-8').splitlines()
				proc = subprocess.run(commandstring, shell=True, check=True, executable='/bin/bash')
				return None

    @staticmethod
		def get_rocm_version():
				torch_hip_version = torch.version.hip
				vers = torch_hip_version.split('.')
				ret_ver = '$rocm_version'
				if len(vers) >= 2:
						ret_ver = vers[0] + '.' + vers[1]
				return ret_ver
