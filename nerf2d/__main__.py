from pathlib import Path

import numpy as np
import torch
import typer
from jaxtyping import install_import_hook

with install_import_hook("nerf2d", "beartype.beartype"):
    from nerf2d import create_dataset
    from nerf2d.image_io import save_image


def main(
    shape_path: Path,
    image_path: Path,
    output_path: Path,
    num_train_views: int = 50,
    num_test_views: int = 20,
    kernel_size: int = 5,
    preview_resolution: int = 512,
    render_resolution: int = 256,
    seed: int = 0,
) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = create_dataset(
        shape_path,
        image_path,
        num_train_views=num_train_views,
        num_test_views=num_test_views,
        kernel_size=kernel_size,
        preview_resolution=preview_resolution,
        render_resolution=render_resolution,
        device=device,
        seed=seed,
    )

    save_image(dataset["overview"], output_path / "overview.png")
    save_image(dataset["occupancy"], output_path / "occupancy.png")
    for tag in ("train", "test"):
        for index, visualization in enumerate(dataset[tag]["visualizations"]):
            save_image(visualization, output_path / tag / f"{index:0>3}.png")

    np.savez(output_path / "dataset", dataset=dataset)


typer.run(main)
