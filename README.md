# lmxrbs-xri

Low-mass X-ray binaries for an X-ray interferometer: GRS 1915+105 (black hole) and Cyg X-2 (neutron star), modelled with [xrimodel](https://github.com/X-ray-interferometry/xrimodel) and observed with [XRISALIS](https://github.com/X-ray-interferometry/XRISALIS).

Both systems are 7-50 µas across, far below the ~190 µas beam of the 10-baseline array bundled with XRISALIS. What the instrument can still measure is time: the donor's reflection echoes the corona's variability with a delay of (r/c)(1 - sin i cos 2πφ), from seconds at superior conjunction of the compact object to 1.5 (Cyg X-2) and 6.5 (GRS 1915+105) minutes at inferior conjunction. This repository asks how well that echo, and with it the size of the binary, can be recovered, and with what exposure.

## Layout

| Path | Contents |
| --- | --- |
| `src/lmxrbs_xri/targets.py` | The two systems: published parameters (with references), spectra, the corona's variability presets; builds `XRBinary` scenes without a donor wind at any phase |
| `src/lmxrbs_xri/images.py` | Model surface brightness on a fine grid (quadtree leaves spread over the pixels they cover), for display |
| `src/lmxrbs_xri/timing.py` | Band light curves, segment-averaged CCFs, the CCF predicted from a model's transfer functions, chunked light-curve simulation through an ARF and RMF |
| `notebooks/01_systems.ipynb` | The systems, their spectra, the donor's reflection and its delay around the orbit, model images and XRISALIS reconstructions (with the model convolved with the dirty beam), response functions, variability, predicted CCFs |
| `tests/` | The CCF recovers an echo and matches its prediction |

Planned: `02_echo_timing` (simulated light curves through the instrument), `03_orbital_imaging` (what the images show), `04_exposures`; the likelihood for the echo is to be designed together.

## Install

```
pip install -e ".[dev]"
```

This pulls xrimodel (with its JAX engine), XRISALIS from its `xrimodel-integration` branch and `xspectrampoline` (XSPEC models: Cyg X-2's compTT, pexmon reflection; GPL-3.0-or-later). Notebooks are committed executed; `notebooks/make_*.py` rebuild them.
