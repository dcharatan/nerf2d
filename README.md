# nerf2d (Dataset Generation)

_A script that generates toy NeRF datasets, except the world is 2D!_

Compare this to regular NeRF datasets below:

|            | **3D World (Regular NeRF)** | **2D World (This Repo)** |
| ---------- | --------------------------- | ------------------------ |
| World      | 3D                          | 2D                       |
| Images     | 2D (`H * W * 3`)            | 1D (`W * 3`)             |
| Extrinsics | 4x4 Matrix                  | 3x3 Matrix               |
| Intrinsics | 3x3 Matrix                  | 2x2 Matrix               |

## Instructions

Create an environment:

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Run the code:

```bash
python3 -m nerf2d.create_dataset <SVG Path> <Image Path> <Output Folder>
```

There are a few SVG and image files you can try out in the `assets` folder. For example:

```bash
python3 -m nerf2d.create_dataset assets/dots.svg assets/flower.png workspace
```

## Creating Your Own Scenes (SVG Files)

To create your own scene, draw some polygons in Inkscape or Illustrator and save them in a square image. Inspect the resulting SVG file to make sure your polygons are actually `<polygon>` tags, because otherwise, this code won't work.
