from pathlib import Path
from xml.dom import minidom

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor


def get_polygon_line_segments(
    path: Path,
    device: torch.device,
) -> Float[Tensor, "line endpoint 2"]:
    """Return all normalized polygon line segment endpoints in an SVG."""
    segments = []

    # Parse the SVG as XML.
    with path.open("r") as f:
        xml = minidom.parse(f)

    # Read the view box.
    (svg,) = xml.getElementsByTagName("svg")
    view_box = svg.getAttribute("viewBox")
    x, y, width, height = np.fromstring(view_box, sep=" ")
    xy = torch.tensor((x, y), dtype=torch.float32, device=device)
    wh = torch.tensor((width, height), dtype=torch.float32, device=device)

    # Read the SVG's polygons.
    for polygon_tag in xml.getElementsByTagName("polygon"):
        # Extract the polygon's points.
        points = polygon_tag.getAttribute("points")
        points = np.fromstring(points, sep=" ")
        points = torch.tensor(points, dtype=torch.float32, device=device)
        points = rearrange(points, "(p xy) -> p xy", xy=2)

        # Convert the polygon's points to a series of line segments. The first and last
        # points of an SVG polygon are repeated, making the approach below possible.
        polygon_segments = torch.stack((points[:-1], points[1:]), dim=1)
        segments.append(polygon_segments)

    return (torch.cat(segments, dim=0) - xy) / wh
