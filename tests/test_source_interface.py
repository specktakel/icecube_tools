import numpy as np

from icecube_tools.source.flux_model import PowerLawFlux, BrokenPowerLawFlux, PowerLawExpCutoffFlux, PowerLawSubexpCutoffFlux
from icecube_tools.source.source_model import PointSource, DiffuseSource


pl_params = (1e-18, 1e5, 2.2, 1e4, 1e8)
bpl_params = (1e-18, 1e5, 2.2, 3.0, 1e4, 1e8)
plec_params = (1e-13, 1e3, -2., 1e3, 1e4, 1e8)
pl_subexp_cutoff_params = (1e-13, 1e3, 2., 1e3, 0.5, 1e4, 1e8)

flux_models = [PowerLawFlux, BrokenPowerLawFlux, PowerLawExpCutoffFlux, PowerLawSubexpCutoffFlux]
params_list = [pl_params, bpl_params, plec_params, pl_subexp_cutoff_params]


def test_flux_models():

    for flux_model, params in zip(flux_models, params_list):

        flux = flux_model(*params)

        # Check boundary
        assert flux.spectrum(1e3) == 0.

        assert flux.spectrum(1e5) != 0.

        assert flux.spectrum(1e9) == 0.

        # Check integration
        if flux_model == PowerLawFlux or flux_model == BrokenPowerLawFlux:
            int_low = flux.integrated_spectrum(1e4, 1e5)

            int_high = flux.integrated_spectrum(1e5, 1e6)

            assert int_low > int_high

        elif flux_model == PowerLawExpCutoffFlux or flux_model == PowerLawSubexpCutoffFlux:
            integral = flux.integrated_spectrum(1e4, 1e6)

            assert integral > 0

        # Check total flux density
        if flux_model == PowerLawFlux:
            assert flux.total_flux_density() > 0

        # Check sampling
        if not flux_model == PowerLawSubexpCutoffFlux:
            samples = flux.sample(1000)

            assert np.all(samples >= 1e4)

            assert np.all(samples <= 1e8)


def test_source_definition():

    flux = flux_models[0](*params_list[0])

    redshift = 0.5

    coord = (0.1, 0.2)

    point_source = PointSource(flux, z=redshift, coord=coord)

    assert point_source.z == redshift

    assert point_source.coord == coord

    diffuse_source = DiffuseSource(flux, z=redshift)

    assert diffuse_source.z == redshift
