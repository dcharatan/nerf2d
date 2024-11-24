import torch
from einops import einsum
from jaxtyping import Float
from torch import Tensor

from nerf2d.scene_extraction import Scene

from .coordinate_conversion import generate_conversions
from .rendering import render_over_image
from .types import Scalar, sanitize_scalar


def draw_scene(
    image: Float[Tensor, "3 height width"],
    scene: Scene,
    width: Scalar,
    num_msaa_passes: int = 1,
) -> Float[Tensor, "3 height width"]:
    device = image.device
    width = sanitize_scalar(width, device)
    (num_lines,) = torch.broadcast_shapes(
        scene.colors.shape[0],
        scene.endpoints.shape[0],
        width.shape,
    )

    # Convert world-space points to pixel space.
    start = scene.endpoints[:, 0]
    end = scene.endpoints[:, 1]
    _, h, w = image.shape
    world_to_pixel, _ = generate_conversions((h, w), device, (-1, 1), (-1, 1))
    start = world_to_pixel(start)
    end = world_to_pixel(end)

    def color_function(
        xy: Float[Tensor, "point 2"],
    ) -> Float[Tensor, "point 4"]:
        # Define a vector between the start and end points.
        delta = end - start
        delta_norm = delta.norm(dim=-1, keepdim=True)
        u_delta = delta / delta_norm

        # Define a vector between each sample and the start point.
        indicator = xy - start[:, None]

        # Determine whether each sample is inside the line in the parallel direction.
        parallel = einsum(u_delta, indicator, "l xy, l s xy -> l s")
        parallel_inside_line = (parallel <= delta_norm) & (parallel > 0)

        # Determine whether each sample is inside the line perpendicularly.
        perpendicular = indicator - parallel[..., None] * u_delta[:, None]
        perpendicular_inside_line = perpendicular.norm(dim=-1) < 0.5 * width[:, None]

        inside_line = parallel_inside_line & perpendicular_inside_line

        # Compute round caps.
        near_start = indicator.norm(dim=-1) < 0.5 * width[:, None]
        inside_line |= near_start
        end_indicator = indicator = xy - end[:, None]
        near_end = end_indicator.norm(dim=-1) < 0.5 * width[:, None]
        inside_line |= near_end

        # Determine the sample's color via linear interpolation of the corresponding
        # line segment's endpoint colors.
        t = (parallel / delta_norm).clip(min=0, max=1)[..., None]
        color = scene.colors[:, 1:] * t + scene.colors[:, :1] * (1 - t)
        arrangement = inside_line * torch.arange(num_lines, device=device)[:, None]
        line_index = arrangement.argmax(dim=0)
        color = color[line_index, torch.arange(xy.shape[0], device=device), :]
        rgba = torch.cat((color, inside_line.any(dim=0).float()[:, None]), dim=-1)

        return rgba

    return render_over_image(image, color_function, device, num_passes=num_msaa_passes)
