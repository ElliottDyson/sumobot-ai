#!/usr/bin/env bash

# This host mounts the NVIDIA driver libraries outside the default dynamic-loader path.
# Do not add CUDA's stubs directory: a stub libcuda can shadow the real driver.
_driver_dir=/usr/lib64-nvidia
_cuda_runtime_dir=/usr/local/cuda/targets/x86_64-linux/lib

if [[ -d "${_driver_dir}" ]]; then
  export LD_LIBRARY_PATH="${_driver_dir}:${_cuda_runtime_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
else
  export LD_LIBRARY_PATH="${_cuda_runtime_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

unset _driver_dir _cuda_runtime_dir
