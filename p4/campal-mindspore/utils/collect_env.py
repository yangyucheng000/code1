"""This file holding some environment constant for sharing by other files."""

import os.path as osp
import subprocess
import sys
from collections import defaultdict
import mindspore



def get_build_config():
    return mindspore.get_context("device_target")


def collect_env():
    """Collect the information of the running environments.

    :return:
        dict: The environment information. The following fields are contained.

            - sys.platform: The variable of ``sys.platform``.
            - Python: Python version.
            # deprecated:
            #   - (deprecated)CUDA available: Bool, indicating if CUDA is available.

            # Add:
            - GPU_available: Bool, indicating if GPU is available.
            - Ascend_available: Bool, indicating if Ascend is available.

            - GPU devices: Device type of each GPU.
            - CUDA_HOME (optional): The env var ``CUDA_HOME``.
            - NVCC (optional): NVCC version.
            - GCC: GCC version, "n/a" if GCC is not installed.
            - PyTorch: PyTorch version.
            - PyTorch compiling details: The output of \
                ``torch.__config__.show()``.
            - TorchVision (optional): TorchVision version.
            - OpenCV: OpenCV version.

    """
    env_info = {'sys.platform': sys.platform, 'Python': sys.version.replace('\n', '')}

    GPU_available = mindspore.context.get_context("device_target") == 'GPU'
    Ascend_available = mindspore.context.get_context("device_target") == 'Ascend'
    env_info['GPU_available'] = GPU_available
    env_info['Ascend_available'] = Ascend_available

    if GPU_available or Ascend_available:

        CUDA_HOME = None

        if CUDA_HOME is not None and osp.isdir(CUDA_HOME):
            try:
                nvcc = osp.join(CUDA_HOME, 'bin/nvcc')
                nvcc = subprocess.check_output(
                    f'"{nvcc}" -V | tail -n1', shell=True)
                nvcc = nvcc.decode('utf-8').strip()
            except subprocess.SubprocessError:
                nvcc = 'Not Available'
            env_info['NVCC'] = nvcc

    try:
        gcc = subprocess.check_output('gcc --version | head -n1', shell=True)
        gcc = gcc.decode('utf-8').strip()
        env_info['GCC'] = gcc
    except subprocess.CalledProcessError:  # gcc is unavailable
        env_info['GCC'] = 'n/a'


    env_info['Mindspore'] = mindspore.get_context("device_target")

    return env_info



def get_dist_info():
    rank = 0
    world_size = 1
    return rank, world_size

