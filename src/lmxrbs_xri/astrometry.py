"""Spectro-astrometry: where in the sky the Fe-band light sits relative to the continuum.

The donor's reflection is a larger share of the light in the Fe band than in the
continuum, so the Fe-band centroid sits slightly towards the donor and follows it around
the orbit. These functions give the model's centroids (truth), simulate the centroid an
observation measures from its photons' fringe phases (``xrisalis.recipes.position_sums``),
and fit the measured offsets around the orbit with the model's orbital pattern plus a
constant (instrumental) offset.
"""

from __future__ import annotations

import copy

import numpy as np
import xrimodel as xm
from xrisalis import generate_photons, redistribute
from xrisalis import recipes as xr
from xrisalis.sources import XrimodelSource


def centroid(model, band, setup=None):
    """The model's detected-count-weighted centroid in the (true-energy) ``band``, keV:
    (East, North) uas, and the count rate (counts/s, ARF summed over baselines)."""
    sx = sy = sr = 0.0
    for layer in model.layers:
        x, y, r = xr.pixel_rates(layer, *band, nbin=12, setup=setup)
        sx, sy, sr = sx + np.sum(r * x), sy + np.sum(r * y), sr + r.sum()
    return sx / sr, sy / sr, sr


def band_limited(model, lo, hi):
    """A copy of ``model`` that emits only photons with rest-frame energies in [lo, hi] keV
    (each layer's spectrum cut there and its flux reduced to the part inside), so a
    simulation of one band does not draw the photons it would discard. Keep a margin of a
    few detector resolutions around the detected band you select."""
    out = copy.deepcopy(model)
    for layer in out.layers:
        if hasattr(layer, "maps"):
            frac = []
            for k, t in enumerate(layer.templates):
                f = np.diff(t.cdf_at(np.array([lo, hi])))[0]
                keep = (t.edges[1:] > lo) & (t.edges[:-1] < hi)
                v = np.where(keep, t.values, 0.0)
                layer.templates[k] = xm.Spectrum(t.edges, v / max(f, 1e-300), t.name)
                frac.append(f)
            layer.maps["NORM"] = layer.maps["NORM"] * np.asarray(frac)[layer.maps["TIDX"].astype(int)]
        else:
            sp = layer.spectrum
            keep = (sp.edges[1:] > lo) & (sp.edges[:-1] < hi)
            layer.spectrum = xm.Spectrum(sp.edges, np.where(keep, sp.values, 0.0), sp.name)
    return out


def measure(model, exposure, band, true_band, setup=None, chunk=2e4, seed=0):
    """Simulate ``exposure`` s of ``model`` through XRISALIS and measure the centroid of the
    photons with detected energies in ``band`` (keV) from their fringe phases, about the
    model's reference position (the compact object). Only photons with true energies in
    ``true_band`` are drawn; the exposure is simulated in chunks of ``chunk`` s whose
    position sums are added. Returns (east, north, cov, n) as
    ``xrisalis.recipes.position_from_sums``."""
    setup = xr.bundled() if setup is None else setup
    m = band_limited(model, *true_band)
    total = None
    t_done, k = 0.0, 0
    while t_done < exposure - 1e-6:
        dt = min(chunk, exposure - t_done)
        ph = generate_photons(XrimodelSource(m), setup.arf, dt, seed=seed + 2 * k)
        ev = redistribute(ph, setup.instrument, setup.rmf, seed=seed + 2 * k + 1)
        s = xr.position_sums(ev, setup, energy_kev=band, phase_center=(model.ra, model.dec))
        total = s if total is None else xr.add_sums(total, s)
        t_done += dt
        k += 1
    return xr.position_from_sums(total)


def fold_fit(offsets, covs, pattern):
    """Fit measured centroid offsets around the orbit, ``offsets`` (n, 2) uas with 2x2
    covariances ``covs``, as a constant offset plus ``s`` times the model's orbital
    ``pattern`` (n, 2): weighted least squares. Returns (s, sigma_s, constant (2,),
    chi2, dof)."""
    a_rows, y, w = [], [], []
    for d, c, p in zip(np.asarray(offsets), np.asarray(covs), np.asarray(pattern)):
        a_rows.append(np.column_stack([np.eye(2), p]))
        y.append(d)
        w.append(np.linalg.inv(c))
    fisher = sum(a.T @ wi @ a for a, wi in zip(a_rows, w))
    rhs = sum(a.T @ wi @ yi for a, wi, yi in zip(a_rows, w, y))
    cov = np.linalg.inv(fisher)
    p = cov @ rhs
    chi2 = float(sum((yi - a @ p) @ wi @ (yi - a @ p) for a, wi, yi in zip(a_rows, w, y)))
    return float(p[2]), float(np.sqrt(cov[2, 2])), p[:2], chi2, 2 * len(y) - 3


def fold_sigma(pattern, rates, exposure_per_phase, sigma1):
    """Expected sigma of ``s`` (the pattern's amplitude, 1 for the model) for a fold fit
    with a free constant, from photon noise alone: ``rates`` counts/s in the band per
    phase, ``sigma1`` the one-axis per-photon precision (uas)."""
    covs = [np.eye(2) * sigma1**2 / (r * exposure_per_phase) for r in rates]
    return fold_fit(np.zeros_like(np.asarray(pattern)), covs, pattern)[1]


__all__ = ["band_limited", "centroid", "fold_fit", "fold_sigma", "measure"]
