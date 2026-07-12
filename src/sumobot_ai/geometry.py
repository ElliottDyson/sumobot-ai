from __future__ import annotations

import torch


def quaternion_rotate_xyzw(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by xyzw quaternions with ordinary PyTorch broadcasting."""
    if quaternion.shape[-1] != 4 or vector.shape[-1] != 3:
        raise ValueError("quaternion/vector trailing dimensions must be four/three")
    xyz = quaternion[..., :3]
    w = quaternion[..., 3:4]
    twice_cross = 2.0 * torch.linalg.cross(xyz, vector, dim=-1)
    return vector + w * twice_cross + torch.linalg.cross(xyz, twice_cross, dim=-1)


def quaternion_rotate_inverse_xyzw(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate world-frame vectors into the corresponding quaternion frame."""
    conjugate = torch.cat((-quaternion[..., :3], quaternion[..., 3:4]), dim=-1)
    return quaternion_rotate_xyzw(conjugate, vector)
