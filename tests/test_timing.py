"""The CCF of simulated light curves: echo lag recovered, and the linear-response
prediction matches the measured CCF within its errors."""

import numpy as np
import xrimodel as xm

from lmxrbs_xri.timing import ccf, centroid_lag, predicted_ccf


def _psd(f):
    return xm.lorentzian_psd(f, 0.3, 0.01)


def test_ccf_echo_lag_and_prediction():
    dt, t_tot, tau = 2.0, 4e5, 40.0
    f = np.geomspace(1e-6, 0.5 / dt, 4000)
    drv = xm.PSDDriver(f, _psd(f), seed=5, transform="linear", dt=0.25).realise(-200.0, t_tot)
    t = np.arange(0.0, t_tot, dt) + 0.5 * dt
    rng = np.random.default_rng(1)
    # x: prompt only; y: a fifth of its rate echoes the driver tau later, plus a constant
    rx, ry_prompt, ry_echo, ry_const = 50.0, 20.0, 5.0, 10.0
    x = rng.poisson(rx * dt * drv(t))
    y = rng.poisson(dt * (ry_prompt * drv(t) + ry_echo * drv(t - tau) + ry_const))
    # short segments and lags up to 80% of them: the prediction must follow the estimator
    lags, c, err, _ = ccf(x, y, n_seg=250, max_lag=200)
    _, pred = predicted_ccf(_psd, (rx, ry_prompt), ([0.0], [ry_echo]), ([0.0], [tau]),
                            (0.0, ry_const), dt, 250, 200)
    # neighbouring lags are strongly correlated, so compare point by point
    assert np.all(np.abs(c - pred) < 4 * err)
    # the echo alone: correlate x with the excess of y over its prompt part
    echo = rng.poisson(dt * ry_echo * drv(t - tau) * 40)
    lags, c, _, _ = ccf(x, echo, n_seg=1000, max_lag=60)
    assert abs(centroid_lag(lags, c) * dt - tau) < 2 * dt
