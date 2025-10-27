"""Utilities."""

from functools import lru_cache
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from prpl_utils.structs import Image
from skimage.transform import resize  # pylint: disable=no-name-in-module


def load_avatar_asset(filename: str) -> Image:
    """Load an image of an avatar."""
    asset_dir = Path(__file__).parent / "assets" / "avatars"
    image_file = asset_dir / filename
    return plt.imread(image_file)


@lru_cache(maxsize=None)
def get_avatar_by_name(avatar_name: str, tilesize: int) -> Image:
    """Helper for rendering."""
    # Assume that the avatar name is saved as the filename with .png.
    im = load_avatar_asset(f"{avatar_name}.png")
    shape = (tilesize, tilesize, 3)
    return resize(im[:, :, :3], shape, preserve_range=True)  # type: ignore


def render_avatar_grid(avatar_grid: NDArray, tilesize: int = 64) -> Image:
    """Helper for rendering."""
    height, width = avatar_grid.shape
    canvas = np.zeros((height * tilesize, width * tilesize, 3))

    for r in range(height):
        for c in range(width):
            avatar_name: str | None = avatar_grid[r, c]
            if avatar_name is None:
                continue
            im = get_avatar_by_name(avatar_name, tilesize)
            canvas[
                r * tilesize : (r + 1) * tilesize,
                c * tilesize : (c + 1) * tilesize,
            ] = im

    return (255 * canvas).astype(np.uint8)
