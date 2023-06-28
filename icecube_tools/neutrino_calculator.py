import numpy as np
from scipy.optimize import fsolve

from .source.source_model import Source, PointSource, DIFFUSE, POINT
from .source.flux_model import FluxModel, PowerLawFlux
from .detector.effective_area import EffectiveArea
from .detector.detector import TimeDependentIceCube
from .cosmology import luminosity_distance, Mpc_to_cm
from .point_source_likelihood.energy_likelihood import MarginalisedIntegratedEnergyLikelihood
from .utils.data import Uptime

"""
Module for calculating the number of neutrinos,
given a flux model and effective area.
"""


M_TO_CM = 100.0
GEV_TO_TEV = 1.0e-3
YEAR_TO_SEC = 3.154e7


class NeutrinoCalculator:
    """
    Calculate the expected number of detected neutrinos.
    """

    def __init__(
        self,
        sources,
        effective_area,
        energy_resolution: MarginalisedIntegratedEnergyLikelihood=None
    ):
        """
        Calculate the expected number of detected neutrinos.

        :param sources: A list of Source instances.
        :param effective_area: An EffectiveArea instance.
        """

        self._sources = sources

        self._effective_area = effective_area
        
        self._energy_resolution = energy_resolution

    @property
    def source(self):

        return self._sources

    @source.setter
    def source(self, value):

        if not isinstance(value, Source):

            raise ValueError(str(value) + " is not an instance of Source")

        else:

            self._source.append(value)

    @property
    def effective_area(self):

        return self._effective_area

    @effective_area.setter
    def effective_area(self, value):

        if not isinstance(value, EffectiveArea):

            raise ValueError(str(value) + " is not an instance of EffectiveArea")

        else:

            self._effective_area = value

    def _diffuse_calculation(self, source):

        Em = self.effective_area.true_energy_bins[:-1]
        EM = self.effective_area.true_energy_bins[1:]
        czm = self.effective_area.cos_zenith_bins[:-1]
        czM = self.effective_area.cos_zenith_bins[1:]

        # Switch labels due to sign
        sdm = -czM.copy()
        sdM = -czm.copy()

        dec_c = np.arcsin((sdm + sdM) / 2)

        # TODO fix this for the data-driven background model
        integrated_spectrum = source.flux_model.integrated_spectrum(Em, EM)
        integrated_direction = (czM - czm) * 2 * np.pi

        bin_c = (np.log10(Em) + np.log10(EM)) / 2
        p_det_above_thr = np.ones((bin_c.size, czm.size))
        if self._energy_resolution is not None:
            for c_e in range(bin_c.size):
                for c_c in range(czm.size):
                    p_det_above_thr[c_e, c_c] = self._energy_resolution.p_det_above_threshold(np.power(10, bin_c[c_e]), dec_c[c_c])

        aeff = self._selected_aeff * M_TO_CM ** 2   # 1st index is energy, 2nd is cosz
        dN_dt = integrated_spectrum.dot(aeff * p_det_above_thr).dot(integrated_direction)

        return dN_dt * self._time * source.redshift_factor

    def _select_single_cos_zenith(self, source):

        # cos(zenith) = -sin(declination)
        cos_zenith = -np.sin(source.coord[1])

        selected_bin_index = (
            np.digitize(cos_zenith, self.effective_area.cos_zenith_bins) - 1
        )

        return selected_bin_index


    def _point_source_calculation(self, source):

        selected_bin_index = self._select_single_cos_zenith(source)

        Em = self.effective_area.true_energy_bins[:-1]
        EM = self.effective_area.true_energy_bins[1:]
        # needs to be replaced by an integral
        # going over reconstructed energy from minimally detected maximally detected
        # or take the hi_nu shortcut:
        # calculate integral at bin center, and assume that's fine
        integrated_flux = source.flux_model.integrated_spectrum(Em, EM)
        bin_c = (np.log10(Em) + np.log10(EM)) / 2
        p_det_above_thr = np.ones((bin_c.size))
        if self._energy_resolution is not None:
            for c in range(bin_c.size):
                p_det_above_thr[c] = self._energy_resolution.p_det_above_threshold(np.power(10, bin_c[c]), source.coord[1])

        aeff = self._selected_aeff.T[selected_bin_index] * M_TO_CM ** 2
        # need to multiply with p(E detected above threshold)
        # threshold given by data releas

        dN_dt = np.dot(aeff * p_det_above_thr, integrated_flux)

        return dN_dt * self._time


    def __call__(self, time=1, min_energy=1e2, max_energy=1e9, min_cosz=-1, max_cosz=1):
        """
        Calculate the number of expected neutrinos,
        taking into account the observation time and
        possible further constraints on the effective
        area as a function of energy and cos(zenith).
        Returns list of expected neutrino numbers for
        each source.

        !! NB: We assume Aeff is zero outside of specified
        energy and cos(zenith)!!

        :param time: Observation time in years.
        :param min_energy: Aeff energy lower bound [GeV].
        :param max_energy: Aeff energy upper bound [GeV].
        :param min_cosz: Aeff cos(zenith) lower bound.
        :param max_cosz: Aeff cos(zenith) upper bound.
        """

        self._time = time * YEAR_TO_SEC  # s

        self._selected_effective_area_values = self.effective_area.values.copy()

        # @TODO: Add contribution from bins on boundary.
        self._selected_effective_area_values[
            self.effective_area.true_energy_bins[1:] < min_energy
        ] = 0
        self._selected_effective_area_values[
            self.effective_area.true_energy_bins[:-1] > max_energy
        ] = 0

        self._selected_effective_area_values.T[
            self.effective_area.cos_zenith_bins[1:] < min_cosz
        ] = 0
        self._selected_effective_area_values.T[
            self.effective_area.cos_zenith_bins[:-1] > max_cosz
        ] = 0

        N = []

        for source in self._sources:

            src_min_energy = source.flux_model._lower_energy
            src_max_energy = source.flux_model._upper_energy

            self._selected_aeff = self._selected_effective_area_values.copy()
            self._selected_aeff[
                self.effective_area.true_energy_bins[1:] < src_min_energy
            ] = 0
            self._selected_aeff[
                self.effective_area.true_energy_bins[:-1] > src_max_energy
            ] = 0

            if source.source_type == DIFFUSE:

                n = self._diffuse_calculation(source)

            elif source.source_type == POINT:

                n = self._point_source_calculation(source)

            else:

                raise ValueError(str(source.source_type) + " is not recognised.")

            N.append(n)

        return N


class PhiSolver:
    """
    For flexible calculation of point source
    fluxes that give a certain number of expected
    neutrinos in a detector.
    """

    def __init__(
        self,
        effective_area,
        Enorm=1e5,
        Emin=1e2,
        Emax=1e9,
        time=1,
        min_cosz=-1.0,
        max_cosz=1.0,
        energy_resolution: MarginalisedIntegratedEnergyLikelihood=None
    ):
        """
        :param effective_area: An EffectiveArea instance
        :param Enorm: Normlisation energy of source spectrum [GeV]
        :param Emin: Minimum energy [GeV]
        :param Emax: Maximum energy [GeV]
        :param min_cosz: Minimum cos(zenith)
        :param max_cosz: Maximum cos(zenith)
        :param time: Observation time [year]
        """

        self._effective_area = effective_area

        self._Enorm = Enorm

        self._Emin = Emin
        self._Emax = Emax

        self._time = time

        self._min_cosz = min_cosz
        self._max_cosz = max_cosz

        if energy_resolution is not None:
            self._energy_resolution = energy_resolution

    def _solve_for_phi(self, phi_norm, Nnu, dec, index):
        """
        For use within get_phi_norm.
        """

        power_law = PowerLawFlux(
            phi_norm,
            self._Enorm,
            index,
            lower_energy=self._Emin,
            upper_energy=self._Emax,
        )
        source = PointSource(flux_model=power_law, coord=(np.pi, np.deg2rad(dec)))
        if hasattr(self, "_energy_resolution"):
            nu_calc = NeutrinoCalculator([source], self._effective_area, self._energy_resolution)
        else:
            nu_calc = NeutrinoCalculator([source], self._effective_area)
        return (
            Nnu
            - nu_calc(
                time=self._time, min_cosz=self._min_cosz, max_cosz=self._max_cosz
            )[0]
        )

    def __call__(self, Nnu, dec, index, guess=1e-19):
        """
        Get equivalent point source flux normalisation
        needed to produce an expected number of neutrinos,
        Nnu, in a detector.

        :param Nnu: Expected number of neutrinos
        :param dec: Declination of point source
        :param index: Spectral index of point source
        """

        phi_norm = fsolve(self._solve_for_phi, x0=guess, args=(Nnu, dec, index))[0]

        return phi_norm
    

class TimeDependentPhiSolver():
    def __init__(self, *data_periods, sources=[], eres_dict={}):
        self._uptime = Uptime(data_periods)
        self._tirf = TimeDependentIceCube.from_periods(self._uptime.irf_periods)
        self._phi_solvers = {}
        for p in self._uptime.irf_periods:
            self._phi_solvers[p] = PhiSolver(
                self._tirf[p].effective_area,
                self._uptime.cumulative_time_obs()[p],
                eres_dict.get(
                    p, MarginalisedIntegratedEnergyLikelihood(
                        p,
                        np.linspace(1, 8, 25)
                    )
                )
            )
        


class zSolver:
    """
    To calculate the redshift corresponding to a flux normalisation,
    for a given L and gamma.
    """

    def __init__(self, Emin):
        """
        :param Emin: The miniminum/normalisation energy [TeV]
        """

        self._Emin = Emin

    def _phi_norm(self, z, L, gamma):

        dl = luminosity_distance(z) * Mpc_to_cm  # cm

        A = L / (4 * np.pi * np.power(dl, 2))  # TeV cm^-2 s^-1
        B = np.power(1 + z, 2 - gamma) * (
            (gamma - 2) / np.power(self._Emin, 2)
        )  # TeV^-2

        return A * B

    def _solve_for_z(self, z, phi_norm, L, gamma):
        """
        :param z: Redshift
        :param phi_norm: Point source flux normalisation at Emin [TeV^-1 cm^-2 s^-1 sr^-1]
        :param L: Source luminosity [TeV s^-1]
        :param gamma: Spectral index
        """

        phi_norm_test = self._phi_norm(z, L, gamma)

        return phi_norm - phi_norm_test

    def __call__(self, phi_norm, L, gamma, guess=0.1):

        z = fsolve(self._solve_for_z, x0=guess, args=(phi_norm, L, gamma))[0]

        return z

    def get_L(self, rate, gamma):
        """
        Find luminosity for a given rate and index.
        This is useful as comparing constant L plots can be
        misleading.

        :param rate: Total rate above Emin [s^-1]
        :param gamma: Spectral index
        """

        A = (gamma - 1) / (gamma - 2)

        B = np.power(self._Emin, 2 - gamma) / np.power(self._Emin, 1 - gamma)

        return rate * A * B
