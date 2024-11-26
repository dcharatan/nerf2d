import torch
from einops import repeat
from jaxtyping import Bool, Float
from torch import Tensor

from nerf2d.scene_extraction import Scene

# This intersection code is based on the following solve:
# from sympy import Eq, Symbol, solve
# r_ox = Symbol("r_ox")
# r_oy = Symbol("r_oy")
# r_dx = Symbol("r_dx")
# r_dy = Symbol("r_dy")
# l_ox = Symbol("l_ox")
# l_oy = Symbol("l_oy")
# l_dx = Symbol("l_dx")
# l_dy = Symbol("l_dy")
# t = Symbol("t")
# u = Symbol("u")
# x_equation = Eq(r_ox + t * r_dx, l_ox + u * l_dx)
# y_equation = Eq(r_oy + t * r_dy, l_oy + u * l_dy)
# solve([x_equation, y_equation], [t, u])

# The solve gives the following:
# {
#     t: (l_dx * l_oy - l_dx * r_oy - l_dy * l_ox + l_dy * r_ox)
#     / (l_dx * r_dy - l_dy * r_dx),
#     u: (-l_ox * r_dy + l_oy * r_dx - r_dx * r_oy + r_dy * r_ox)
#     / (l_dx * r_dy - l_dy * r_dx),
# }


def intersect(
    origins: Float[Tensor, "ray 2"],
    directions: Float[Tensor, "ray 2"],
    scene: Scene,
    background: Float[Tensor, "rgb=3"],
    eps: float = 1e-7,
) -> Float[Tensor, "ray 3"]:
    # This definitely isn't the most efficient approach in PyTorch, but it's the fastest
    # way to implement the solve above.
    r_ox, r_oy = origins[:, None].unbind(dim=-1)
    r_dx, r_dy = directions[:, None].unbind(dim=-1)
    start, end = scene.endpoints.unbind(dim=1)
    l_ox, l_oy = start[None, :].unbind(dim=-1)
    l_dx, l_dy = (end - start)[None, :].unbind(dim=-1)

    t_numerator = l_dx * l_oy - l_dx * r_oy - l_dy * l_ox + l_dy * r_ox
    u_numerator = -l_ox * r_dy + l_oy * r_dx - r_dx * r_oy + r_dy * r_ox
    denominator = l_dx * r_dy - l_dy * r_dx

    t = t_numerator / denominator
    u = u_numerator / denominator

    # Pick the closest hit.
    parallel = denominator.abs() < eps
    hit = (t > 0) & (u >= 0) & (u <= 1)
    valid = hit & ~parallel
    t[~valid] = torch.inf
    closest_hit = t.argmin(dim=-1)

    # Compute color for the closest hit.
    segments = torch.arange(t.shape[0], device=origins.device)
    mix = u[segments, closest_hit][:, None]
    start_color, end_color = scene.colors.unbind(dim=1)
    color = (1 - mix) * start_color[closest_hit] + mix * end_color[closest_hit]

    # Return the color if there was actually a hit. Else, return the background.
    return torch.where(valid.any(dim=-1)[:, None], color, background)


def render_occupancy(
    scene: Scene,
    eps: float = 1e-7,
    resolution: int = 512,
) -> Bool[Tensor, "height width"]:
    device = scene.endpoints.device

    # Define parallel rays going right.
    r_oy = (torch.arange(resolution, device=device) + 0.5) / resolution * 2 - 1
    r_oy = r_oy[:, None]
    r_ox = -torch.ones_like(r_oy)
    r_dy = torch.zeros_like(r_oy)
    r_dx = torch.ones_like(r_oy)

    # Copy-pasted from above.
    start, end = scene.endpoints.unbind(dim=1)
    l_ox, l_oy = start[None, :].unbind(dim=-1)
    l_dx, l_dy = (end - start)[None, :].unbind(dim=-1)

    t_numerator = l_dx * l_oy - l_dx * r_oy - l_dy * l_ox + l_dy * r_ox
    u_numerator = -l_ox * r_dy + l_oy * r_dx - r_dx * r_oy + r_dy * r_ox
    denominator = l_dx * r_dy - l_dy * r_dx

    t = t_numerator / denominator
    u = u_numerator / denominator

    # Compute t at hit points.
    parallel = denominator.abs() < eps
    hit = (t > 0) & (u >= 0) & (u <= 1)
    valid = hit & ~parallel
    t[~valid] = torch.inf

    # Compute occupancy using the winding number. Note: r_oy + 1 gives [0, 2] instead of
    # [-1, 1], which is what we want in order to compare to t.
    # TODO: One must technically detect corners to count them twice for the winding
    # number, but for toy experiment stuff, that probably doesn't matter...
    t_sorted = t.sort(dim=-1)[0]
    grid = repeat(r_oy + 1, "w () -> h w", h=resolution)
    indices = torch.searchsorted(t_sorted.contiguous(), grid.contiguous())
    return (indices % 2).type(torch.bool)
