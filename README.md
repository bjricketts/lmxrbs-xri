# lmxrbs-xri

Low-mass X-ray binaries for an X-ray interferometer: GRS 1915+105 (black hole) and Cyg X-2 (neutron star), modelled with [xrimodel](https://github.com/X-ray-interferometry/xrimodel) and observed with [XRISALIS](https://github.com/X-ray-interferometry/XRISALIS).

Both systems are 7-50 µas across, far below the ~190 µas beam of the 10-baseline array bundled with XRISALIS. What the instrument can still measure is time: the donor's reflection echoes the corona's variability with a delay of (r/c)(1 - sin i cos 2πφ), from seconds at superior conjunction of the compact object to 1.5 (Cyg X-2) and 6.5 (GRS 1915+105) minutes at inferior conjunction. The instrument also measures where photons come from: the donor's reflection is a larger share of the Fe-band light, so the Fe-band centroid follows the donor around the orbit (spectro-astrometry). This repository asks how well the echo and the centroid shift, and with them the size of the binary, can be recovered, and with what exposure.

## Layout

| Path | Contents |
| --- | --- |
| `src/lmxrbs_xri/targets.py` | The two systems: published parameters (with references), spectra, the corona's variability presets; builds `XRBinary` scenes without a donor wind at any phase |
| `src/lmxrbs_xri/astrometry.py` | Spectro-astrometry: model centroids per band, band-limited models, a simulated centroid measurement through XRISALIS from photon fringe phases, the folded fit (orbital pattern plus constant offset) |
| `src/lmxrbs_xri/images.py` | Model surface brightness on a fine grid (quadtree leaves spread over the pixels they cover), for display |
| `src/lmxrbs_xri/timing.py` | Band light curves, segment-averaged CCFs, the CCF predicted from a model's transfer functions, chunked light-curve simulation through an ARF and RMF |
| `notebooks/01_systems.ipynb` | The systems, their spectra, the donor's reflection and its delay around the orbit, model images and XRISALIS reconstructions (with the model convolved with the dirty beam), response functions, variability, predicted CCFs |
| `notebooks/02_astrometry.ipynb` | Per-photon position precision by band; the Fe-band minus continuum centroid around the orbit; forecast and a simulated campaign of eight 100 ks visits, fitted |
| `scripts/simulate_campaign.py` | The campaign for notebook 2 (about four hours; writes `data/campaign_<target>.npz`) |
| `tests/` | The CCF recovers an echo and matches its prediction; band-limited models keep their band; the folded fit recovers a known amplitude |

Planned: echo timing with simulated light curves, exposures; the likelihoods are to be designed together.

## Install

```
pip install -e ".[dev]"
```

This pulls xrimodel (with its JAX engine), XRISALIS from its `xrimodel-integration` branch and `xspectrampoline` (XSPEC models: Cyg X-2's compTT, pexmon reflection; GPL-3.0-or-later). Notebooks are committed executed; `notebooks/make_*.py` rebuild them.
