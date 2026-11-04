"""Band-limited models and the folded fit."""

import numpy as np
import xrimodel as xm
from xrisalis import recipes

from lmxrbs_xri import CYGX2
from lmxrbs_xri.astrometry import band_limited, fold_fit


def test_band_limited_model_keeps_the_band_and_drops_the_rest():
    m = CYGX2.scene(0.5, grid=xm.RegularGrid(nx=12, ny=12, scale=3.0)).render()
    cut = band_limited(m, 5.6, 7.3)
    for lo, hi, same in ((5.9, 7.0, True), (1.0, 4.0, False)):
        full = sum(recipes.pixel_rates(L, lo, hi)[2].sum() for L in m.layers)
        kept = sum(recipes.pixel_rates(L, lo, hi)[2].sum() for L in cut.layers)
        assert (abs(kept / full - 1) < 1e-3) if same else (kept < 1e-6 * full)


def test_fold_fit_recovers_amplitude_and_offset():
    rng = np.random.default_rng(3)
    ph = np.arange(8) / 8
    pattern = np.column_stack([np.cos(2 * np.pi * ph), 0.5 * np.sin(2 * np.pi * ph)])
    fits = []
    for _ in range(400):
        sig = 0.3
        off = (0.1, -0.2) + 0.7 * pattern + rng.normal(0, sig, pattern.shape)
        fits.append(fold_fit(off, [np.eye(2) * sig**2] * 8, pattern))
    s = np.array([f[0] for f in fits])
    ds = fits[0][1]
    assert abs(s.mean() - 0.7) < 4 * ds / np.sqrt(len(s))
    assert 0.85 < s.std() / ds < 1.15
    assert abs(np.mean([f[3] for f in fits]) / fits[0][4] - 1) < 0.1
