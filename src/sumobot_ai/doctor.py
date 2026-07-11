from __future__ import annotations

import ctypes
import importlib.util
import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    ok: bool
    detail: str


def run_checks() -> list[Check]:
    checks: list[Check] = []
    loader_path = os.environ.get("LD_LIBRARY_PATH", "")
    checks.append(
        Check(
            "NVIDIA driver path",
            "/usr/lib64-nvidia" in loader_path,
            "LD_LIBRARY_PATH includes /usr/lib64-nvidia"
            if "/usr/lib64-nvidia" in loader_path
            else "source scripts/activate_gpu.sh",
        )
    )
    for library in ("libcuda.so.1", "libnvidia-ml.so.1", "libcudart.so.12"):
        try:
            ctypes.CDLL(library)
        except OSError as error:
            checks.append(Check(library, False, str(error)))
        else:
            checks.append(Check(library, True, "loadable"))
    modules = ("torch", "warp", "newton", "mujoco", "mujoco_warp", "tensordict")
    for module in modules:
        available = importlib.util.find_spec(module) is not None
        required = module == "torch"
        detail = "installed" if available else ("missing" if required else "optional extra not installed")
        checks.append(Check(f"Python {module}", available or not required, detail))
    try:
        import torch

        available = torch.cuda.is_available()
        detail = torch.cuda.get_device_name(0) if available else "torch.cuda.is_available() is false"
        checks.append(Check("PyTorch CUDA", available, detail))
    except Exception as error:  # pragma: no cover - diagnostic must report rather than crash
        checks.append(Check("PyTorch CUDA", False, repr(error)))
    return checks
