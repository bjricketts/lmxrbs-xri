"""Simulate a phase-folded spectro-astrometry campaign through XRISALIS.

For each target: visits of ``--visit`` s centred on ``--n-phases`` orbital phases. Each
visit is simulated in sub-chunks rendered at their own phase (the orbit moves during a
visit), with only Fe-band photons drawn; their fringe-phase position sums are added. The
continuum (reference) band is not simulated in full (it has ~50 times the photons):
its per-photon precision is measured on a short stretch and its error added in the fit.

Writes data/campaign_<target>.npz. Usage:

    python scripts/simulate_campaign.py [--targets grs cx2] [--visit 1e5] [--n-phases 8]
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
from xrisalis import generate_photons, redistribute
from xrisalis import recipes as xr
from xrisalis.sources import XrimodelSource

from lmxrbs_xri import TARGETS
from lmxrbs_xri.astrometry import band_limited, centroid

FE, FE_TRUE = (5.9, 7.0), (5.6, 7.3)
CONT = {"grs": ((3.0, 5.0), (2.7, 5.3)), "cx2": ((1.0, 4.0), (0.8, 4.3))}
DATA = Path(__file__).resolve().parents[1] / "data"
warnings.filterwarnings("ignore")


def sums(model, band, true_band, exposure, setup, seed):
    ph = generate_photons(XrimodelSource(band_limited(model, *true_band)), setup.arf, exposure,
                          seed=seed)
    ev = redistribute(ph, setup.instrument, setup.rmf, seed=seed + 1)
    return xr.position_sums(ev, setup, energy_kev=band, phase_center=(model.ra, model.dec))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["grs", "cx2"])
    ap.add_argument("--visit", type=float, default=1e5)
    ap.add_argument("--n-phases", type=int, default=8)
    ap.add_argument("--chunk", type=float, default=1.25e4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    setup = xr.bundled()
    DATA.mkdir(exist_ok=True)
    for name in args.targets:
        t = TARGETS[name]
        cont, cont_true = CONT[name]
        visit_phase = args.visit / (t.period_days * 86400.0)
        phases = np.arange(args.n_phases) / args.n_phases
        out = DATA / f"campaign_{name}.npz"
        # the continuum band's per-photon precision, from a short stretch at phase 0
        m = t.scene(0.0).render()
        _, _, c_cov, c_n = xr.position_from_sums(sums(m, cont, cont_true, 200.0, setup, 999))
        c_rate = centroid(m, cont)[2]
        rows = []
        if out.exists():  # resume: keep the visits already simulated with these settings
            old = np.load(out)
            if float(old["visit"]) == args.visit and int(old["seed"]) == args.seed:
                rows = [tuple(r) for r in old["rows"]]
        for i, ph0 in enumerate(phases):
            if any(abs(r[0] - ph0) < 1e-9 for r in rows):
                continue
            t0 = time.time()
            n_chunks = int(np.ceil(args.visit / args.chunk))
            total, fe_c, cont_c, rate = None, [], [], []
            for k in range(n_chunks):
                ph = ph0 + visit_phase * ((k + 0.5) / n_chunks - 0.5)
                m = t.scene(ph % 1.0).render()
                s = sums(m, FE, FE_TRUE, args.visit / n_chunks, setup,
                         seed=args.seed + 1000 * i + 2 * k)
                total = s if total is None else xr.add_sums(total, s)
                fe_c.append(centroid(m, FE)[:2])
                cont_c.append(centroid(m, cont)[:2])
                rate.append(centroid(m, FE)[2])
            east, north, cov, n = xr.position_from_sums(total)
            rows.append((ph0, east, north, *cov.ravel(), n, *np.mean(fe_c, 0),
                         *np.mean(cont_c, 0), np.mean(rate)))
            print(f"{name} phase {ph0:.3f}: Fe centroid ({east:.3f}, {north:.3f}) "
                  f"+- {np.sqrt(cov[0, 0]):.3f} uas from {n} photons; model "
                  f"({np.mean(fe_c, 0)[0]:.3f}, {np.mean(fe_c, 0)[1]:.3f}); "
                  f"{time.time() - t0:.0f} s", flush=True)
            # saved after every visit, so an interrupted run resumes where it stopped
            np.savez(out, rows=np.array(sorted(rows)), visit=args.visit, seed=args.seed,
                     cont_sigma1=np.sqrt(np.diag(c_cov) * c_n), cont_rate=c_rate,
                     columns=np.array(["phase", "east", "north", "c_ee", "c_en", "c_ne", "c_nn",
                                       "n", "fe_east", "fe_north", "cont_east", "cont_north",
                                       "fe_rate"]))
        print(f"{name}: {len(rows)} visits in {out}", flush=True)


if __name__ == "__main__":
    main()
