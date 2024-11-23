from pathlib import Path

import torch
import typer
from einops import repeat
from jaxtyping import install_import_hook

with install_import_hook("nerf2d", "beartype.beartype"):
    from nerf2d.drawing.cameras import draw_cameras
    from nerf2d.drawing.scene import draw_scene
    from nerf2d.geometry.random_extrinsics import generate_random_extrinsics
    from nerf2d.image_io import save_image
    from nerf2d.scene_extraction import extract_scene


def main(
    shape_path: Path,
    image_path: Path,
    output_path: Path,
    num_train_views: int = 50,
    num_test_views: int = 20,
    kernel_size: int = 5,
    preview_resolution: int = 512,
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

    # Render images for the scene.
    for tag, num_images in (("train", num_train_views), ("test", num_test_views)):
        # Sample camera parameters.
        extrinsics = generate_random_extrinsics(num_images, 2, 0.9)
        intrinsics = torch.eye(2, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "i j -> n i j", n=num_images).clone()
        intrinsics[:, :-1, -1] = 0.5
        intrinsics[:, :-1, :-1] = 0.75

        # Save the images.
        for index, (c2w, k) in enumerate(zip(extrinsics, intrinsics)):
            overview_with_camera = draw_cameras(overview, c2w[None], k[None], 1, 2)
            save_image(
                overview_with_camera, output_path / tag / f"cameras/{index:0>3}.png"
            )

        a = 1

    print("hello")


if __name__ == "__main__":
    typer.run(main)
