from typing import Optional

import numpy as np
import torch
from einops import repeat
from jaxtyping import Float
from scipy.stats import special_ortho_group as so_group
from torch import Tensor


def generate_random_extrinsics(
    n: int,
    dimensionality: int,
    radius: float,
    generator: Optional[np.random.Generator] = None,
) -> Float[Tensor, "number dim_homogeneous dim_homogeneous"]:
    d = dimensionality
    rotation = [
        torch.tensor(so_group.rvs(d, random_state=generator), dtype=torch.float32)
        for _ in range(n)
    ]
    rotation = torch.stack(rotation)
    translation = -radius * rotation[:, :, -1]
    extrinsics = repeat(torch.eye(d + 1, dtype=torch.float32), "i j -> n i j", n=n)
    extrinsics = extrinsics.clone()
    extrinsics[:, :-1, :-1] = rotation
    extrinsics[:, :-1, -1] = translation
    return extrinsics
