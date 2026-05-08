from icecube_tools.detector.effective_area import EffectiveArea
from icecube_tools.detector.energy_resolution import EnergyResolution
from icecube_tools.detector.angular_resolution import AngularResolution
from icecube_tools.detector.detector import IceCube
from icecube_tools.source.flux_model import PowerLawFlux
from icecube_tools.source.source_model import DiffuseSource, PointSource
from icecube_tools.utils.data import SimEvents, I3_10, I3_14
from icecube_tools.detector.r2021 import R2021IRF
from icecube_tools.simulator import Simulator, TimeDependentSimulator
from icecube_tools.point_source_likelihood.spatial_likelihood import (
    EventDependentSpatialGaussianLikelihood,
)
from icecube_tools.point_source_likelihood.energy_likelihood import (
    MarginalisedEnergyLikelihood2021,
    MarginalisedIntegratedEnergyLikelihood,
)
from icecube_tools.point_source_likelihood.point_source_likelihood import (
    PointSourceLikelihood,
    TimeDependentPointSourceLikelihood,
)

import h5py

import numpy as np
from os.path import join
from scipy.stats import rv_histogram
import matplotlib.pyplot as plt
import pytest

aeff = EffectiveArea.from_dataset(period="IC86_II")
irf = R2021IRF.from_period("IC86_II")
detector = IceCube.from_dataset(period="IC86_II")


@pytest.fixture
def source_coords():
    return (np.pi, np.deg2rad(30))


@pytest.fixture
def sources(source_coords):
    diff_flux_norm = 3e-21  # Flux normalisation in units of GeV^-1 cm^-2 s^-1 sr^-1
    point_flux_norm = 5e-19  # Flux normalisation in units of GeV^-1 cm^-2 s^-1
    norm_energy = 1e5  # Energy of normalisation in units of GeV
    min_energy = 1e2  # GeV
    max_energy = 1e8  # GeV

    diff_power_law = PowerLawFlux(
        diff_flux_norm, norm_energy, 3.7, min_energy, max_energy
    )
    diff_source = DiffuseSource(diff_power_law, z=0.0)

    point_power_law = PowerLawFlux(
        point_flux_norm, norm_energy, 2.5, min_energy, max_energy
    )
    point_source = PointSource(point_power_law, z=0.0, coord=source_coords)
    s = [diff_source, point_source]
    return s


@pytest.fixture
def single_file_sim(output_directory, random_seed, sources):

    # Set up simulation
    simulator = Simulator(sources, detector, period="IC86_II")
    simulator.time = 1  # year

    # Run simulation
    simulator.run(show_progress=True, seed=random_seed)

    # Save to file
    simulator.write_to_h5(join(output_directory, "sim_output_86_II.h5"), sources)


def test_fit_from_single(source_coords, output_directory, single_file_sim):

    events = SimEvents.load_from_h5(join(output_directory, "sim_output_86_II.h5"))

    new_reco_bins = np.linspace(2, 9, num=25)

    energy_likelihood = {
        "IC86_II": MarginalisedIntegratedEnergyLikelihood(
            "IC86_II", I3_10, new_reco_bins
        )
    }
    spatial_likelihood = EventDependentSpatialGaussianLikelihood()

    source_coord = (np.pi, np.deg2rad(30))
    likelihood = PointSourceLikelihood(
        spatial_likelihood,
        energy_likelihood["IC86_II"],
        events.ra["IC86_II"],
        events.dec["IC86_II"],
        events.reco_energy["IC86_II"],
        events.ang_err["IC86_II"],
        source_coords,
    )
    energy_likelihood["IC86_II"]._min_index = 1.4
    energy_likelihood["IC86_II"]._max_index = 4.0
    likelihood.get_test_statistic()

    # Again with time dependent detector, should yield the same values!
    tllh = TimeDependentPointSourceLikelihood(
        source_coord,
        ["IC86_II"],
        events._ra,
        events._dec,
        events._reco_energy,
        events._ang_err,
        energy_likelihood,
        times={"IC86_II": 1},
    )

    m = tllh._minimize()

    assert likelihood._best_fit_index == pytest.approx(m.values["index"], abs=0.1)

    assert likelihood._best_fit_ns == pytest.approx(m.values["ns"], abs=0.1)


# def test_fit_from_multi(output_directory, random_seed, sources):
#    pass
