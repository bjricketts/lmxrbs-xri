"""The two systems, GRS 1915+105 (black hole) and Cyg X-2 (neutron star), as xrimodel scenes.

Each :class:`Target` holds the published system parameters (with references), the
time-averaged spectrum and the corona's variability, and builds an
:class:`xrimodel.engine.XRBinary` without a donor wind at any orbital phase. Phase 0 is
superior conjunction of the compact object (behind the donor).

Components (the ``XRBinary`` layers):

* ``<name>_corona``: the illuminating point source at the compact object, driven by the
  variability model (``<name>_var``);
* ``<name>_disc_bb``: the thermal disc plus the broad (inner-disc) Fe line, unresolved,
  constant, and not illuminating anything;
* ``<name>_star`` and ``<name>_disc``: the corona's light reflected by the donor and by the
  outer disc, following the corona's light curve with their light-travel delays.

Simplifications, beyond those of ``XRBinary``: the disc's thermal emission neither varies
nor illuminates the donor; the prompt broad Fe line is constant; Cyg X-2's Comptonised
boundary layer (compTT) is not a power law, so its reflection uses xrimodel's approximate
template rather than pexmon.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import xrimodel as xm
from xrimodel.engine import Scene, XRBinary, build_spectrum

RSUN_CM = 6.957e10


def _scaled(spec, flux, lo=0.5, hi=10.0, edges=None):
    """A copy of spectrum spec ``spec`` scaled to energy flux ``flux`` (erg/s/cm^2) in
    [lo, hi] keV (unabsorbed)."""
    edges = xm.energy_grid(0.1, 20.0, 0.01) if edges is None else edges
    f = build_spectrum(spec, edges).energy_flux(lo, hi)
    out = dict(spec)
    out["norm"] = spec.get("norm", 1.0) * flux / f
    return out


@dataclass(frozen=True)
class Variability:
    """The corona's power spectrum as a sum of shapes (fractional rms^2/Hz), realised from
    ``f_min`` to ``f_max`` Hz. ``lorentzians``: (rms, hwhm, centre); ``powerlaws``: (rms,
    index, f_lo, f_hi), rms over [f_lo, f_hi]."""

    lorentzians: tuple = ()
    powerlaws: tuple = ()
    f_min: float = 1e-5
    f_max: float = 4.0
    note: str = ""

    def frequency(self, n=2000):
        return np.geomspace(self.f_min, self.f_max, n)

    def psd(self, f):
        f = np.asarray(f, float)
        p = np.zeros_like(f)
        for rms, hwhm, centre in self.lorentzians:
            p += xm.lorentzian_psd(f, rms, hwhm, centre)
        for rms, index, lo, hi in self.powerlaws:
            p += xm.powerlaw_psd(f, rms, index, lo, hi)
        return np.where((f >= self.f_min) & (f <= self.f_max), p, 0.0)

    def rms(self):
        f = self.frequency(20000)
        return float(np.sqrt(np.trapezoid(self.psd(f), f)))

    def driver(self, name, seed=0, span=None):
        f = self.frequency()
        return xm.PSDDriver(f, self.psd(f), name=name, seed=seed, span=span)


@dataclass(frozen=True)
class Target:
    name: str
    label: str
    ra: float  # deg
    dec: float
    distance_pc: float
    m_compact: float  # Msun
    m_donor: float
    period_days: float
    inclination: float  # deg
    node_pa: float  # deg East of North of the +X node (see XRBinary)
    nh_ism: float  # cm^-2
    corona: dict  # spectrum spec, unabsorbed
    disc: dict  # spectrum spec: thermal disc + prompt broad Fe line, unabsorbed
    variability: dict  # preset name -> Variability
    r_disc_in_rg: float = 1e4
    r_disc_out_rg: float = 1e6
    disc_h_over_r: float = 0.1
    roche_fill: float = 0.99  # donor's volume-equivalent radius / its Roche lobe
    refs: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ geometry

    def binary(self, phase=0.0, variability=None, **kw) -> XRBinary:
        """The system at ``phase`` (no wind). ``variability`` names the corona's driver
        (``<name>_var``) if given; any XRBinary option overrides."""
        b = XRBinary(self.name, self.corona, m_donor=self.m_donor, m_compact=self.m_compact,
                     period_days=self.period_days, inclination=self.inclination,
                     node_pa=self.node_pa, phase=phase, disc_spectrum=self.disc, wind=False,
                     r_disc_in_rg=self.r_disc_in_rg, r_disc_out_rg=self.r_disc_out_rg,
                     disc_h_over_r=self.disc_h_over_r, nh_ism=self.nh_ism,
                     driver=None if variability is None else f"{self.name}_var")
        b.r_donor = self.roche_fill * b.roche_lobe("donor") * b.separation_cm() / RSUN_CM
        for k, v in kw.items():
            setattr(b, k, v)
        return b

    def scales(self):
        """Angular and light-travel scales: separation a (cm, uas, light-s), the donor's
        radius and the compact object's Roche lobe (uas, r_g), the disc's radii (uas)."""
        b = self.binary()
        a = b.separation_cm()
        uas = a / (self.distance_pc * 3.0856775814913673e18) * 206264.806e6
        return dict(a_cm=a, a_uas=uas, a_light_s=a / 2.99792458e10,
                    r_donor_uas=b.roche_lobe("donor") * self.roche_fill * uas,
                    roche_lobe_compact_uas=b.roche_lobe("compact") * uas,
                    roche_lobe_compact_rg=b.roche_lobe_rg(),
                    disc_out_uas=self.r_disc_out_rg * b.rg_cm() / a * uas,
                    donor_offset_uas=uas * self.m_compact / (self.m_compact + self.m_donor))

    def donor_sky(self, phase):
        """The donor's centre on the sky at ``phase``: (East, North) offset in uas from the
        compact object."""
        b = self.binary(phase)
        f = b.frame()
        u_a, u_b = b.sky_axes()
        d = f["donor"] - f["bh"]
        uas_per_cm = self.scales()["a_uas"] / f["a"]
        return (d @ f["e_a"] * u_a + d @ f["e_b"] * u_b) * uas_per_cm

    def grid(self, phase=0.0, donor_cells=24, max_level=14):
        """A quadtree for ``phase``: refined towards the compact object (where the disc's
        reflection is concentrated), over the disc (to ~32 leaves across its radius) and
        over the donor (to ~``donor_cells`` leaves across its radius)."""
        s = self.scales()
        root = 2 * 1.15 * (s["donor_offset_uas"] + s["r_donor_uas"])
        r_d, r_disc = s["r_donor_uas"], s["disc_out_uas"]
        dx, dy = self.donor_sky(phase)

        def refine(x, y, size):
            near_co = (np.hypot(x, y) < 4 * size + 0.02 * root) & (size > root / 2**11)
            near_donor = np.hypot(x - dx, y - dy) < 1.3 * r_d + size
            on_disc = (np.hypot(x, y) < r_disc + size) & (size > r_disc / 32)
            return near_co | on_disc | (near_donor & (size > r_d / donor_cells))

        return xm.QuadtreeGrid.refine(root, refine, max_level=max_level, min_level=5)

    def scene(self, phase=0.0, variability=None, seed=0, span=None, grid=None, **kw):
        """A Scene with the system at ``phase`` and, if ``variability`` names a preset, the
        corona's driver (realised with ``seed`` over ``span``)."""
        sc = Scene(ra=self.ra, dec=self.dec, distance_pc=self.distance_pc,
                   grid=self.grid(phase) if grid is None else grid)
        if variability is not None:
            sc.add_driver(self.variability[variability].driver(f"{self.name}_var", seed, span))
        sc.add(self.binary(phase, variability, **kw))
        return sc

    def with_(self, **kw) -> Target:
        return replace(self, **kw)


# --------------------------------------------------------------------------- GRS 1915+105

# Bright state: AstroSat/NICER, Oct 2017, high-flux segment (Dhaka et al. 2024,
# arXiv:2408.02875): tbabs N_H = 4.2e22, thcomp*diskbb with T_in = 1.84 keV, norm 137,
# tau = 0.5, kT_e = 100 keV and 40% of the disc photons scattered, and a broad Fe line at
# 6.6 keV (sigma 0.6 keV, 0.012 ph/s/cm^2). Evaluated with XSPEC's thcomp, the Comptonised
# part has Gamma = 2.37 over 4-12 keV and 1.06e-8 erg/s/cm^2 in 2-10 keV (unabsorbed): the
# corona is that power law (it overestimates the Comptonised photons below ~2 keV, which
# N_H = 4e22 hides from us but which do reach the donor).
_GRS_DISC_BB = {"kind": "diskbb", "t_in": 1.84, "norm": 137.0 * (1 - 0.4)}
_GRS_CORONA = _scaled({"kind": "powerlaw", "gamma": 2.37}, 1.06e-8, lo=2.0, hi=10.0)

GRS1915 = Target(
    name="grs", label="GRS 1915+105",
    ra=288.798114, dec=10.945751,  # Reid et al. 2014 (VLBA)
    distance_pc=9400.0,  # 3-D kinematic distance, Reid & Miller-Jones 2023
    m_compact=11.2,  # Reid & Miller-Jones 2023 (i = 64 deg)
    m_donor=0.47,  # Steeghs et al. 2013
    period_days=33.85,  # Steeghs et al. 2013
    inclination=64.0,  # Reid & Miller-Jones 2023 (jet; orbit assumed aligned)
    node_pa=50.0,  # approaching jet at PA ~140 deg (Reid et al. 2014, Table 2): node line _|_
    nh_ism=4.2e22,
    corona=_GRS_CORONA,
    disc={"kind": "sum", "parts": [_GRS_DISC_BB,
                                   {"kind": "gaussian_line", "e0": 6.6, "sigma": 0.6,
                                    "flux": 0.012}]},
    variability={
        "broadband": Variability(lorentzians=((0.15, 0.05, 0.0), (0.10, 2.0, 0.0)),
                                 note="two zero-centred Lorentzians, 18% rms in total"),
    },
    r_disc_out_rg=2.0e6,  # ~0.8 of the black hole's Roche lobe (2.5e6 r_g)
    refs={"system": "Steeghs et al. 2013; Reid et al. 2014; Reid & Miller-Jones 2023",
          "spectrum": "Dhaka et al. 2024 (arXiv:2408.02875), high-flux segment"},
)

# --------------------------------------------------------------------------- Cyg X-2

# NICER, Oct 2017 (Zhang et al. 2025, arXiv:2506.13503, HBO orbit of obs 1): tbfeo
# N_H = 2.4e21; diskbb T_in = 0.59 keV with 7.6e-9 erg/s/cm^2; compTT (kT_0 = 0.86, kT_e = 3,
# tau = 4.2) with 22.2e-9 erg/s/cm^2; broad Fe line at 6.4 keV (sigma 1.0 keV, 3e-10
# erg/s/cm^2); fluxes 0.5-10 keV. compTT needs XSPEC models (xspectrampoline).
_CX2_CORONA = _scaled({"kind": "xspec", "model": "C_compTT", "params": [0.0, 0.86, 3.0, 4.2, 1.0]},
                      22.2e-9)
def _line(e0, sigma, energy_flux, lo=0.5, hi=10.0):
    """A Gaussian line spec with the photon flux that gives ``energy_flux`` in [lo, hi]."""
    f1 = build_spectrum({"kind": "gaussian_line", "e0": e0, "sigma": sigma, "flux": 1.0},
                        xm.energy_grid(0.1, 20.0, 0.01)).energy_flux(lo, hi)
    return {"kind": "gaussian_line", "e0": e0, "sigma": sigma, "flux": energy_flux / f1}


_CX2_DISC = {"kind": "sum", "parts": [_scaled({"kind": "diskbb", "t_in": 0.59}, 7.6e-9),
                                      _line(6.4, 1.0, 0.3e-9)]}

CYGX2 = Target(
    name="cx2", label="Cyg X-2",
    ra=326.171500, dec=38.321417,
    distance_pc=7200.0,  # Orosz & Kuulkers 1999 (7-11 kpc in the literature)
    m_compact=1.71,  # Casares et al. 2010
    m_donor=0.60,  # Orosz & Kuulkers 1999
    period_days=9.8444,  # Casares et al. 1998
    inclination=62.5,  # Orosz & Kuulkers 1999
    node_pa=0.0,  # unknown
    nh_ism=2.4e21,
    corona=_CX2_CORONA,
    disc=_CX2_DISC,
    variability={
        # Wijnands et al. 1997 (Ginga): very-low-frequency noise on the lower normal
        # branch/soft vertex, rms over 0.001-1 Hz; realised only above 1 mHz (its measured
        # range: slower changes are motion along the Z track)
        "high": Variability(powerlaws=((0.045, 1.6, 1e-3, 1.0),), f_min=1e-3,
                            note="high intensity level: strong, steep VLFN, no NBO"),
        "medium": Variability(powerlaws=((0.01, 1.0, 1e-3, 1.0),),
                              lorentzians=((0.015, 1.5, 5.5),), f_min=1e-3,
                              note="medium intensity level: weak, flat VLFN and a 5.5 Hz NBO"),
    },
    r_disc_out_rg=2.6e6,  # ~0.8 of the neutron star's Roche lobe (3.3e6 r_g)
    refs={"system": "Casares et al. 1998, 2010; Orosz & Kuulkers 1999",
          "spectrum": "Zhang et al. 2025 (arXiv:2506.13503), HBO orbit of obs 1",
          "variability": "Wijnands et al. 1997; Hasinger & van der Klis 1989"},
)

TARGETS = {t.name: t for t in (GRS1915, CYGX2)}

__all__ = ["CYGX2", "GRS1915", "TARGETS", "Target", "Variability"]
