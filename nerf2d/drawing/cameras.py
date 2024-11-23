import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from nerf2d.geometry.projection import get_world_rays

from .lines import draw_lines
from .types import Scalar, Vector


def draw_cameras(
    image: Float[Tensor, "3 height width"],
    extrinsics: Float[Tensor, "batch 3 3"],
    intrinsics: Float[Tensor, "batch 2 2"],
    color: Vector,
    width: Scalar,
    num_msaa_passes: int = 1,
) -> Float[Tensor, "3 height width"]:
    device = image.device

    # Convert the camera metadata to line segments.
    origins, directions = get_world_rays(
        torch.tensor([0, 1], dtype=torch.float32, device=device)[:, None],
        extrinsics[:, None],
        intrinsics[:, None],
    )
    start = rearrange(origins, "b e xy -> (b e) xy")
    end = rearrange(origins + 10 * directions, "b e xy -> (b e) xy")

    return draw_lines(
        image,
        start,
        end,
        color,
        width,
        "round",
        num_msaa_passes=num_msaa_passes,
        x_range=(-1, 1),
        y_range=(-1, 1),
    )
