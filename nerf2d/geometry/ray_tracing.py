import torch
from jaxtyping import Float
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
