from pathlib import Path

import torch
import typer
from jaxtyping import install_import_hook

with install_import_hook("nerf2d", "beartype.beartype"):
    from nerf2d.drawing.scene import draw_scene
    from nerf2d.image_io import save_image
    from nerf2d.scene_extraction import extract_scene


def main(
    shape_path: Path,
    image_path: Path,
    num_train_views: int = 50,
    num_test_views: int = 20,
    kernel_size: int = 5,
) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    scene = extract_scene(shape_path, image_path, kernel_size, device)

    image = torch.zeros((3, 512, 512), dtype=torch.float32, device=device)
    image = draw_scene(image, scene, 2)

    save_image(image, "test.png")

    print("hello")


if __name__ == "__main__":
    typer.run(main)
