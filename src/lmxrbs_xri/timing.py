"""Light curves, cross-correlation functions and their prediction from a model.

Conventions: light curves are counts per bin on a common, evenly spaced time grid. The
cross-correlation ``ccf(x, y)`` at lag k correlates x(t) with y(t + k dt), so a positive
lag means y lags x: with x the reference (continuum) band and y the band rich in reflected
light, the donor's echo appears at positive lags.

Errors come from averaging over independent segments (the standard approach for evenly
sampled X-ray light curves); the segment length sets the lowest frequency that contributes,
as each segment's mean is removed.
"""

from __future__ import annotations

import numpy as np
import xrimodel as xm

# --------------------------------------------------------------------------- light curves


def band_lightcurves(times, energies, bands, dt, t0, t1):
    """Counts per ``dt`` bin in each energy band (keV, ``[(lo, hi), ...]``) for photons or
    events with arrival ``times`` (s) and ``energies`` (keV) in [t0, t1). Returns (t, counts)
    with counts of shape (n_bands, n_bins) and t the bin starts."""
    edges = np.arange(t0, t1 + 0.5 * dt, dt)
    out = np.empty((len(bands), edges.size - 1))
    for k, (lo, hi) in enumerate(bands):
        sel = (energies >= lo) & (energies < hi)
        out[k] = np.histogram(times[sel], edges)[0]
    return edges[:-1], out


# --------------------------------------------------------------------------- CCF


def _segments(x, n_seg):
    n = x.size // n_seg
    return x[: n * n_seg].reshape(n, n_seg)


def ccf(x, y, n_seg, max_lag):
    """Segment-averaged cross-correlation coefficient of light curves ``x`` and ``y``
    (counts per bin) at lags -max_lag..max_lag bins, from segments of ``n_seg`` bins: the
    cross-covariance averaged over mean-subtracted segments (each lag's sum divided by the
    number of overlapping bins), over the square root of the product of the mean
    variances. Returns (lags in bins,
    mean CCF, standard error of the mean over segments, number of segments)."""
    xs, ys = _segments(np.asarray(x, float), n_seg), _segments(np.asarray(y, float), n_seg)
    xs = xs - xs.mean(axis=1, keepdims=True)
    ys = ys - ys.mean(axis=1, keepdims=True)
    nfft = 1 << int(np.ceil(np.log2(2 * n_seg)))
    cross = np.fft.irfft(np.conj(np.fft.rfft(xs, nfft)) * np.fft.rfft(ys, nfft), nfft)
    lags = np.arange(-max_lag, max_lag + 1)
    cov = cross[:, lags % nfft] / (n_seg - np.abs(lags))[None, :]
    norm = np.sqrt(np.mean(xs**2) * np.mean(ys**2))
    m = cov.shape[0]
    return lags, cov.mean(axis=0) / norm, cov.std(axis=0, ddof=1) / np.sqrt(m) / norm, m


def centroid_lag(lags, c, frac=0.8):
    """Centroid of the CCF above ``frac`` of its peak, in the contiguous region around the
    peak (as for the interpolated CCF in reverberation mapping, Peterson et al. 1998)."""
    i = int(np.argmax(c))
    lo = i
    while lo > 0 and c[lo - 1] >= frac * c[i]:
        lo -= 1
    hi = i
    while hi < c.size - 1 and c[hi + 1] >= frac * c[i]:
        hi += 1
    w = c[lo:hi + 1]
    return float(np.sum(w * lags[lo:hi + 1]) / np.sum(w))


# --------------------------------------------------------------------------- prediction


def predicted_ccf(psd, rate_prompt, responses, delays, rate_const, dt, n_seg, max_lag):
    """The CCF that :func:`ccf` should measure for a linear response to one driver, for two
    bands (x, y).

    ``psd(f)``: the driver's power spectrum (fractional rms^2/Hz). Per band b:
    ``rate_prompt[b]`` counts/s following the driver without delay, ``responses[b]``
    counts/s following it with ``delays`` (s; one array per band, or shared), and
    ``rate_const[b]`` counts/s that do not vary (they add Poisson noise only). The light
    curve is then r_b(t) = rate_prompt D(t) + sum_j responses_j D(t - delay_j) + const with
    D the driver (mean 1). Binning by ``dt`` is a sinc^2(f dt) filter. The covariances are
    then averaged exactly as :func:`ccf` averages them: within segments of ``n_seg`` bins,
    each with its own mean removed, over the overlapping bins at each lag. Returns (lags in
    bins, CCF)."""
    n = int(n_seg)
    n_fft = 8 * n
    f = np.fft.rfftfreq(n_fft, dt)[1:]
    df = 1.0 / (n_fft * dt)
    p = psd(f) * np.sinc(f * dt) ** 2
    resp, mean = [], []
    for b in range(2):
        d = np.asarray(delays[b] if isinstance(delays, (list, tuple)) else delays, float)
        r = np.asarray(responses[b], float)
        resp.append(rate_prompt[b] + np.sum(r[None, :] * np.exp(-2j * np.pi * f[:, None]
                                                                 * d[None, :]), axis=1))
        mean.append(rate_prompt[b] + r.sum() + rate_const[b])
    m = np.arange(-(n - 1), n)

    def cov(spec):  # covariance at lags m * dt, (counts/s)^2
        return np.real(np.exp(2j * np.pi * np.outer(m * dt, f)) @ (spec * p)) * df

    def centred(c):  # E[(x_i - xbar)(y_j - ybar)] for the Toeplitz matrix c(j - i)
        idx = np.arange(n)
        mat = c[(idx[None, :] - idx[:, None]) + n - 1]
        return mat - mat.mean(0, keepdims=True) - mat.mean(1, keepdims=True) + mat.mean()

    cxy = centred(cov(np.conj(resp[0]) * resp[1]))
    lags = np.arange(-max_lag, max_lag + 1)
    num = np.array([np.trace(cxy, offset=k) / (n - abs(k)) for k in lags])
    var = [np.trace(centred(cov(np.abs(resp[b]) ** 2))) / n + mean[b] / dt * (n - 1) / n
           for b in range(2)]
    return lags, num / np.sqrt(var[0] * var[1])


def band_rates(model, band, area_fn, delay_edges, n_e=40):
    """Counts/s per layer group in the energy ``band`` (keV) for ``model``: the driven
    layers' transfer functions summed over energy with the effective area ``area_fn(E)``
    (cm^2), split into the prompt part (delay < first ``delay_edges`` bin width) and the
    delayed response per delay bin, plus the constant (undriven) rate. Delays within the
    first bin count as prompt, so start ``delay_edges`` with a narrow bin. Returns
    (prompt, response per delay bin, delay bin centres, constant)."""
    e = np.linspace(band[0], band[1], n_e + 1)
    area = area_fn(0.5 * (e[1:] + e[:-1]))
    resp = np.zeros(delay_edges.size - 1)
    const = 0.0
    for layer in model.layers:
        if hasattr(layer, "maps"):
            psi = xm.transfer_function(layer, e, delay_edges) * area[:, None]
            r = psi.sum(axis=0)
        else:  # separable point layers: no delay
            sp = layer.spectrum
            cum = np.concatenate([[0.0], np.cumsum(sp.values * sp.widths)])
            r = np.zeros(delay_edges.size - 1)
            r[0] = float(np.sum(np.diff(np.interp(e, sp.edges, cum)) * area))
        if layer.driver is None:
            const += r.sum()
        else:
            resp += r
    centres = 0.5 * (delay_edges[1:] + delay_edges[:-1])
    return resp[0], np.concatenate([[0.0], resp[1:]]), centres, const





# --------------------------------------------------------------------------- simulation


def simulate_lightcurves(model, bands, dt, t_total, arf, rmf, chunk=1000.0, seed=0):
    """Band light curves (counts per ``dt``) of ``model`` observed for ``t_total`` s through
    an ARF (summed over baselines) and an RMF: xrimodel draws the photons in chunks of
    ``chunk`` s, they are thinned by the ARF and given detected energies by the RMF, and
    only the light curves are kept, so long exposures fit in memory. Bands select detected
    energies. The model's drivers must be realised over a fixed ``span`` covering
    [-max delay, t_total] so every chunk sees the same light curve. No fringes or
    attitude: for timing, the interferometer is a photon counter. Returns (t, counts)."""
    from xrisalis.simulate import redistribute_rmf

    rng = np.random.default_rng(seed)
    e_mid, a_tot = np.asarray(arf.energy_mid, float), np.asarray(arf.total, float)
    a_max = float(a_tot.max())
    edges = np.arange(0.0, t_total + 0.5 * dt, dt)
    counts = np.zeros((len(bands), edges.size - 1))
    for t0 in np.arange(0.0, t_total, chunk):
        t1 = min(t0 + chunk, t_total)
        ph = xm.sample_photons(model, t0, t1, area=a_max, rng=rng)
        keep = rng.random(len(ph)) * a_max < np.interp(ph.energy, e_mid, a_tot, left=0, right=0)
        _, det, valid = redistribute_rmf(ph.energy[keep], rmf, rng=rng)
        t = ph.time[keep][valid]
        for k, (lo, hi) in enumerate(bands):
            sel = (det >= lo) & (det < hi)
            counts[k] += np.histogram(t[sel], edges)[0]
    return edges[:-1], counts


__all__ = ["band_lightcurves", "band_rates", "ccf", "centroid_lag", "predicted_ccf",
           "simulate_lightcurves"]
