import numpy as np
from pytest import approx

from icecube_tools.detector.effective_area import EffectiveArea
from icecube_tools.detector.energy_resolution import EnergyResolution
from icecube_tools.detector.angular_resolution import AngularResolution
from icecube_tools.detector.r2021 import R2021IRF
from icecube_tools.detector.detector import IceCube
from icecube_tools.source.flux_model import PowerLawFlux, BrokenPowerLawFlux, PowerLawExpCutoffFlux, PowerLawSubexpCutoffFlux
from icecube_tools.source.source_model import DiffuseSource, PointSource
from icecube_tools.neutrino_calculator import NeutrinoCalculator, PhiSolver
from icecube_tools.simulator import Simulator, TimeDependentSimulator



aeff = EffectiveArea.from_dataset("20181018")
angres = AngularResolution.from_dataset("20181018")
eres = EnergyResolution.from_dataset("20150820")
detector = IceCube(aeff, eres, angres)

pl_params = (1e-18, 1e5, 2.2, 1e4, 1e8)
pl_flux = PowerLawFlux(*pl_params)
point_source = PointSource(pl_flux, z=0.3)
diffuse_source = DiffuseSource(pl_flux)

bpl_params = (1e-18, 1e5, 2.2, 3.0, 1e4, 1e8)
bpl_flux = BrokenPowerLawFlux(*bpl_params)
point_source_bpl = PointSource(bpl_flux, z=0.0)
diffuse_source_bpl = DiffuseSource(bpl_flux)

plec_params = (1e-13, 1e3, -2., 1e3, 1e4, 1e8)
plec_flux = PowerLawExpCutoffFlux(*plec_params)
point_source_plec = PointSource(plec_flux, z=0.0)
diffuse_source_plec = DiffuseSource(plec_flux)

pl_subexp_cutoff_params = (1e-13, 1e3, 2., 1e3, 0.5, 1e4, 1e8)
pl_subexp_cutoff_flux = PowerLawSubexpCutoffFlux(*pl_subexp_cutoff_params)
point_source_pl_subexp_cutoff = PointSource(pl_subexp_cutoff_flux, z=0.0)
diffuse_source_pl_subexp_cutoff = DiffuseSource(pl_subexp_cutoff_flux)

sources = [point_source, diffuse_source, point_source_bpl, diffuse_source_bpl, point_source_plec, 
           diffuse_source_plec, point_source_pl_subexp_cutoff, diffuse_source_pl_subexp_cutoff]


def test_nu_calc():

    nu_calc = NeutrinoCalculator([point_source, point_source_bpl, point_source_plec, point_source_pl_subexp_cutoff], aeff)

    Nnu = nu_calc(
        time=1,
        min_energy=1e4,
        max_energy=1e8,
        min_cosz=-1,
        max_cosz=1,
    )

    assert all(n > 0 for n in Nnu)

    assert all(~np.isnan(n) for n in Nnu)

    phi_solver = PhiSolver(aeff, 1e5, 1e4, 1e8, time=1, min_cosz=-1, max_cosz=1)

    phi_norm = phi_solver(Nnu=Nnu[0], dec=0.0, index=2.2)

    assert phi_norm == approx(1e-18, rel=0.01)


def test_simulation():

    simulator = Simulator(sources, IceCube.from_period("IC86_II"), "IC86_II")
    simulator.time = 1

    simulator.run(seed=42)

    assert min(simulator._true_energy["IC86_II"]) >= 1e4


def test_new_simulation():
    tsim = TimeDependentSimulator(["IC86_I", "IC86_II"], sources)
    tsim.run(seed=42)
