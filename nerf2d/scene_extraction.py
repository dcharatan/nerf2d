from dataclasses import dataclass
from pathlib import Path

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor
from torch.nn.functional import grid_sample
from torchvision.transforms.functional import gaussian_blur

from nerf2d.image_io import load_image
from nerf2d.svg import get_polygon_line_segments


@dataclass
class Scene:
    endpoints: Float[Tensor, "line endpoint=2 xy=2"]
    colors: Float[Tensor, "line endpoint=2 rgb=3"]


def extract_scene(
    shape_path: Path,
    image_path: Path,
    kernel_size: int,
    device: torch.device,
) -> Scene:
    # Load the scene's line segments.
    endpoints = get_polygon_line_segments(shape_path, device)

    # Sample colors for the line segments' endpoints.
    image = load_image(image_path, device)
    image = gaussian_blur(image, kernel_size)
    colors = grid_sample(
        image[None],
        rearrange(endpoints, "l e xy -> () l e xy") * 2 - 1,
        mode="bilinear",
        align_corners=False,
    )
    colors = rearrange(colors, "() rgb l e -> l e rgb")

    return Scene(endpoints, colors)
