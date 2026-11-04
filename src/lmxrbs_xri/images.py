"""Model images for display: surface brightness of xrimodel layers on a fine regular grid.

``xrisalis.recipes.model_image`` puts each layer pixel into the grid pixel holding its
centre, which is right for convolving with a dirty beam on ~15 uas pixels. On a zoomed grid
finer than the model's coarser quadtree leaves that leaves gaps, so here each leaf is
spread over the grid pixels it covers.
"""

from __future__ import annotations

import numpy as np
from xrisalis.recipes import pixel_rates


def surface_brightness(model, grid, layers=None, lo=0.5, hi=12.0, setup=None):
    """Detected count rate per uas^2 (counts/s/uas^2, through the ARF summed over
    baselines) of the layers passing ``layers(name)`` on ``grid`` (a ``RegularGrid``,
    column index increasing towards the West). Point layers go to the pixel holding them."""
    img = np.zeros((grid.ny, grid.nx))
    s = grid.scale
    for layer in model.layers:
        if layers is not None and not layers(layer.name):
            continue
        x, y, r = pixel_rates(layer, lo, hi, setup=setup)
        g = getattr(layer, "grid", None)
        size = g.sizes() if g is not None and g.n > 1 else np.zeros(r.size)
        # pixel-index bounds of each leaf (half-open), at least one pixel
        c0 = np.floor((grid.x0 - (x + size / 2)) / s + grid.nx / 2)
        c1 = np.maximum(np.ceil((grid.x0 - (x - size / 2)) / s + grid.nx / 2), c0 + 1)
        r0 = np.floor((y - size / 2 - grid.y0) / s + grid.ny / 2)
        r1 = np.maximum(np.ceil((y + size / 2 - grid.y0) / s + grid.ny / 2), r0 + 1)
        for k in np.flatnonzero(r > 0):
            a0, a1 = int(max(c0[k], 0)), int(min(c1[k], grid.nx))
            b0, b1 = int(max(r0[k], 0)), int(min(r1[k], grid.ny))
            if a1 > a0 and b1 > b0:
                img[b0:b1, a0:a1] += r[k] / ((c1[k] - c0[k]) * (r1[k] - r0[k]))
    return img / s**2


__all__ = ["surface_brightness"]
