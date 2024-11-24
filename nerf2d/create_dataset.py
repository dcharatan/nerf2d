from pathlib import Path

import numpy as np
import torch
import typer
from einops import repeat
from jaxtyping import install_import_hook
from tqdm import tqdm

with install_import_hook("nerf2d", "beartype.beartype"):
    from nerf2d.drawing.cameras import draw_cameras
    from nerf2d.drawing.scene import draw_scene
    from nerf2d.geometry.projection import get_world_rays, sample_image_grid
    from nerf2d.geometry.random_extrinsics import generate_random_extrinsics
    from nerf2d.geometry.ray_tracing import intersect
    from nerf2d.image_io import save_image
    from nerf2d.scene_extraction import extract_scene
    from nerf2d.visualization.layout import vcat


def main(
    shape_path: Path,
    image_path: Path,
    output_path: Path,
    num_train_views: int = 50,
    num_test_views: int = 20,
    kernel_size: int = 5,
    preview_resolution: int = 512,
    render_resolution: int = 256,
) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_path.mkdir(exist_ok=True, parents=True)

    # Extract the scene from the given shape (SVG file) and image.
    scene = extract_scene(shape_path, image_path, kernel_size, device)

    # Render a preview of the scene.
    r = preview_resolution
    overview = torch.zeros((3, r, r), dtype=torch.float32, device=device)
    overview = draw_scene(overview, scene, 2)
    save_image(overview, output_path / "scene.png")

    dataset = {}

    # Render images for the scene.
    background = torch.zeros((3,), dtype=torch.float32, device=device)
    for tag, num_images in (("train", num_train_views), ("test", num_test_views)):
        # Sample camera parameters.
        extrinsics = generate_random_extrinsics(num_images, 2, 0.9)
        intrinsics = torch.eye(2, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "i j -> n i j", n=num_images).clone()
        intrinsics[:, :-1, -1] = 0.5
        intrinsics[:, :-1, :-1] = 0.75

        # We flip the intrinsics to make the image orientation vs. the top-down view
        # more intuitive.
        intrinsics[:, 0, 0] *= -1

        # Render from the camera parameters.
        images = []
        visualizations = []
        for index, (c2w, k) in enumerate(zip(tqdm(extrinsics, desc=tag), intrinsics)):
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
            save_image(visualization, output_path / tag / f"{index:0>3}.png")
        images = np.stack(images)
        visualizations = np.stack(visualizations)

        dataset[tag] = {
            "images": images,
            "visualizations": visualizations,
            "extrinsics": extrinsics.cpu().numpy(),
            "intrinsics": intrinsics.cpu().numpy(),
        }

    np.savez(output_path / "dataset", dataset=dataset)


if __name__ == "__main__":
    typer.run(main)
