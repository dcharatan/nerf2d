from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from einops import repeat
from jaxtyping import Bool, Float
from tqdm import tqdm

from nerf2d.drawing.cameras import draw_cameras
from nerf2d.drawing.scene import draw_scene
from nerf2d.geometry.projection import get_world_rays, sample_image_grid
from nerf2d.geometry.random_extrinsics import generate_random_extrinsics
from nerf2d.geometry.ray_tracing import intersect, render_occupancy
from nerf2d.scene_extraction import extract_scene
from nerf2d.visualization.layout import vcat


class DatasetSplit(TypedDict):
    extrinsics: Float[np.ndarray, "_ 3 3"]
    intrinsics: Float[np.ndarray, "_ 2 2"]
    images: Float[np.ndarray, "_ 3 render_width"]
    visualizations: Float[np.ndarray, "_ 3 vis_height vis_width"]


class Dataset(TypedDict):
    train: DatasetSplit
    test: DatasetSplit
    overview: Float[np.ndarray, "3 preview_height preview_width"]
    occupancy: Bool[np.ndarray, "preview_height preview_width"]


def create_dataset(
    shape_path: Path,
    image_path: Path,
    num_train_views: int = 50,
    num_test_views: int = 20,
    kernel_size: int = 5,
    preview_resolution: int = 512,
    render_resolution: int = 256,
    device: torch.device = torch.device("cpu"),
    seed: int = 0,
) -> Dataset:
    generator = np.random.default_rng(seed)

    # Extract the scene from the given shape (SVG file) and image.
    scene = extract_scene(shape_path, image_path, kernel_size, device)

    # Render a preview of the scene.
    r = preview_resolution
    overview = torch.zeros((3, r, r), dtype=torch.float32, device=device)
    overview = draw_scene(overview, scene, 2)
    dataset = {}

    # Render images for the scene.
    background = torch.zeros((3,), dtype=torch.float32, device=device)
    for tag, num_images in (("train", num_train_views), ("test", num_test_views)):
        # Sample camera parameters.
        extrinsics = generate_random_extrinsics(
            num_images,
            2,
            0.9,
            generator=generator,
            device=device,
        )
        intrinsics = torch.eye(2, dtype=torch.float32, device=device)
        intrinsics = repeat(intrinsics, "i j -> n i j", n=num_images).clone()
        intrinsics[:, :-1, -1] = 0.5
        intrinsics[:, :-1, :-1] = 0.75

        # We flip the intrinsics to make the image orientation vs. the top-down view
        # more intuitive.
        intrinsics[:, 0, 0] *= -1

        # Render from the camera parameters.
        images = []
        visualizations = []
        for c2w, k in zip(tqdm(extrinsics, desc=tag), intrinsics):
            # Render.
            x, _ = sample_image_grid((render_resolution,), device)
            origins, directions = get_world_rays(x, c2w, k)
            rendered = intersect(origins, directions, scene, background)
            images.append(rendered.cpu().numpy())

            # Save an overview image.
            rendered_vis = repeat(rendered, "w c -> c h w", h=preview_resolution // 16)
            rendered_vis = rendered_vis.cpu().numpy()
            overview_with_camera = draw_cameras(overview, c2w[None], k[None], 0.5, 2)
            overview_with_camera = overview_with_camera.cpu().numpy()
            visualization = vcat(
                (overview_with_camera, rendered_vis), "center", color=1, border=16
            )
            visualizations.append(visualization)
        images = np.stack(images)
        visualizations = np.stack(visualizations)

        dataset[tag] = {
            "images": images,
            "visualizations": visualizations,
            "extrinsics": extrinsics.cpu().numpy(),
            "intrinsics": intrinsics.cpu().numpy(),
        }

    occupancy = render_occupancy(scene, resolution=preview_resolution)
    dataset["occupancy"] = occupancy.cpu().numpy()
    dataset["overview"] = overview.cpu().numpy()

    return dataset
