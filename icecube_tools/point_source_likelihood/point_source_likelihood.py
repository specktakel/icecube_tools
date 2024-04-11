import numpy as np
from iminuit import Minuit
import logging

from .energy_likelihood import *
from .spatial_likelihood import *
from .datadriven_background_likelihood import DataDrivenBackgroundLikelihood

from ..source.source_model import PointSource
from ..source.flux_model import PowerLawFlux
from ..neutrino_calculator import NeutrinoCalculator

# from ..detector.detector import TimeDependentIceCube
from ..utils.data import Uptime

from typing import Dict, List, Tuple, Sequence
from collections import OrderedDict

"""
Module to compute the IceCube point source likelihood
using publicly available information.

Based on the method described in:
Braun, J. et al., 2008. Methods for point source analysis 
in high energy neutrino telescopes. Astroparticle Physics, 
29(4), pp.299–305.

Currently well-defined for searches with
Northern sky muon neutrinos.
"""

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class PointSourceLikelihood:
    """
    Calculate the point source likelihood for a given
    neutrino dataset - in terms of reconstructed
    energies and arrival directions.
    Based on what is described in Braun+2008 and
    Aartsen+2018.
    """

    def __init__(
        self,
        direction_likelihood: SpatialLikelihood,
        energy_likelihood: MarginalisedEnergyLikelihood,
        ras: Sequence[float],
        decs: Sequence[float],
        energies: Sequence[float],
        ang_errs: Sequence[float],
        source_coord: Tuple[float, float],
        which: str = "both",
        vary_atmo: bool = False,
        vary_astro: bool = False,
        bg_energy_likelihood=None,
        bg_spatial_likelihood=None,
        index_prior=None,
        band_width_factor: float = 5.0,
        cosz_bins: np.ndarray = None,
    ):
        """
        Calculate the point source likelihood for a given
        neutrino dataset - in terms of reconstructed
        energies and arrival directions.

        :param direction_likelihood: An instance of SpatialLikelihood.
        :param energy_likelihood: An instance of MarginalisedEnergyLikelihood.
        :param ras: Right ascensions of events, in rad
        :param decs: Declination of events, in rad
        :param energies: The reconstructed nu energies in GeV.
        :param ang_errs: $1 \sigma$ angular errors of events, in degrees
        :param source_coord: (ra, dec) pf the point to test.
        :param which: str, either `spatial`, `both`, which information of event to be used
        :param vary_atmo: bool, if atmospheric background index should be varied, defaults to False
        :param vary_astro: bool, if astroph. background index should be varied, defaults to False, only used if vary_atmo==True
        :param bg_energy_likelihood: Optional energy likelihood for background events,
            has only energy dependence.
        :param index_prior: Optional prior on the spectral index, instance of Prior.
        :param band_width_factor: Optional factor for the minimum angular resolution (largest angles)
            to be considered for the event selection nearby the source, defaults to 3.
        :param cosz_bins: np.ndarray, used to select declination-band-wise events, can be omitted if integrated energylikelihood is used.
        """

        if which not in ["both", "energy", "spatial"]:
            raise ValueError("No other type of likelihood available.")
        else:
            self.which = which
            logger.debug(f"Using {which} likelihoods.")

        self._vary_atmo = vary_atmo

        self._vary_astro = vary_astro

        self._direction_likelihood = direction_likelihood

        self._cosz_bins = cosz_bins

        self._energy_likelihood = energy_likelihood

        self._bg_energy_likelihood = bg_energy_likelihood

        self._bg_spatial_likelihood = bg_spatial_likelihood

        if isinstance(
            self._direction_likelihood, EnergyDependentSpatialGaussianLikelihood
        ):

            self._band_width = (
                band_width_factor * self._direction_likelihood.get_low_res()
            )

        else:

            self._band_width = (
                band_width_factor * self._direction_likelihood._sigma
            )  # degrees

        self._ras = ras

        self._decs = decs

        self._energies = energies

        self._index_prior = index_prior

        self._ang_errs = ang_errs

        self.source_coord = source_coord  # moved select_nearby_events into setter

        self._angular_distance = self.angular_distance()

        # Sensible values based on Braun+2008
        # and Aartsen+2018 analyses
        self._bg_index = 3.7
        self._ns_min = 0.0

        try:
            self._max_index = self._energy_likelihood._max_index
            self._min_index = self._energy_likelihood._min_index
        except AttributeError:
            self._max_index = 3.95

        # Can't have more source events than actual events...
        self._ns_max = self.N

        self.Ntot = len(self._energies)

    def angular_distance(self):
        src_ra = self.source_coord[0]
        src_dec = self.source_coord[1]
        ra = self._selected_ras
        dec = self._selected_decs
        cos_r = np.cos(src_ra - ra) * np.cos(src_dec) * np.cos(dec) + np.sin(
            src_dec
        ) * np.sin(dec)

        # Handle possible floating precision errors.
        idx = np.nonzero((cos_r < -1.0))
        cos_r[idx] = 1.0
        idx = np.nonzero((cos_r > 1.0))
        cos_r[idx] = 1.0

        r = np.arccos(cos_r)
        return r

    @property
    def source_coord(self):
        return self._source_coord

    @source_coord.setter
    def source_coord(self, new_coord):
        """
        Sets new source coordinates (ra, dec) and updates event selection.
        :param new_coord: New coordinate tuple
        """

        self._source_coord = new_coord

        if (
            isinstance(self._energy_likelihood, MarginalisedIntegratedEnergyLikelihood)
            and not self._cosz_bins
        ):
            dec_bins = np.arcsin(-self._energy_likelihood._aeff.cos_zenith_bins)

        elif self._cosz_bins is not None:
            dec_bins = np.arcsin(-self._cosz_bins)

        else:
            raise ValueError("No cosz bins provided")

        dec_bins.sort()
        zero_dec_idx = np.digitize(0.0, dec_bins) - 1

        # How many dec bins away is self._band_width at the equator? Take as conservative number of dec bins to consider
        upper_dec_idx = np.digitize(np.deg2rad(self._band_width), dec_bins) - 1
        num_of_bins = upper_dec_idx - zero_dec_idx

        dec = new_coord[1]
        # Includes a symmetric number of bins below and above the declination in the source selection,
        # sources ON the bin edge are not considered but treated the way np.digitize handles it.
        # self._band_width should be large enough anyways
        dec_idx = np.digitize(dec, dec_bins) - 1
        dec_idx_low = dec_idx - num_of_bins
        dec_idx_high = dec_idx + num_of_bins + 1

        # Catch exceptions for sources close to the North pole or South pole
        if dec_idx_high >= dec_bins.size:
            dec_idx_high = dec_bins.size - 1
        if dec_idx_low < 0:
            dec_idx_low = 0
        self._dec_low = dec_bins[dec_idx_low]
        self._dec_high = dec_bins[dec_idx_high]

        if self._dec_low < np.arcsin(-1.0) or np.isnan(self._dec_low):
            self._dec_low = np.arcsin(-1.0)

        if self._dec_high > np.arcsin(1.0) or np.isnan(self._dec_high):
            self._dec_high = np.arcsin(1.0)

        self._band_solid_angle = 4 * np.pi
        # self._band_solid_angle = (
        #    2 * np.pi * (np.sin(self._dec_high) - np.sin(self._dec_low))
        # )

        # Two pathological cases to consider here:
        # RA is just below 2pi, then self._ra_high will spill over to 2pi >> RA > 0
        # RA is just above 0, then self._ra_low will spill over to 2pi > RA >> 0
        # Is taken care of in `_select_nearby_events()`
        self._ra_low = self.source_coord[0] - np.deg2rad(self._band_width)
        self._ra_high = self.source_coord[0] + np.deg2rad(self._band_width)

        self._select_nearby_events()

    def update_events(self, ra, dec, reco_energy, ang_err):
        """
        Provide new events and call `self._select_nearby_events()`
        """

        self._ras = ra
        self._decs = dec
        self._energies = reco_energy
        self._ang_errs = ang_err
        self._select_nearby_events()

    def _select_nearby_events(self):
        """
        Select events used in analysis nearby the source.
        """

        if self._ra_low < 0.0:
            selected = np.nonzero(
                (
                    (self._decs >= self._dec_low)
                    & (self._decs <= self._dec_high)
                    & (
                        ((self._ras >= 0.0) & (self._ras <= self._ra_high))
                        # include all events that are close to the source
                        # from the `other side of 2pi´ someone call a mathematician, how do you properly say that?
                        # 2pi ambiguity?
                        | (
                            (self._ras >= self._ra_low + 2 * np.pi)
                            & (self._ras <= 2 * np.pi)
                        )
                    )
                )
            )
        elif self._ra_high > 2 * np.pi:
            selected = np.nonzero(
                (
                    (self._decs >= self._dec_low)
                    & (self._decs <= self._dec_high)
                    & (
                        ((self._ras <= 2 * np.pi) & (self._ras >= self._ra_low))
                        | (
                            (self._ras >= 0.0)
                            & (self._ras <= self._ra_high - 2 * np.pi)
                        )
                    )
                )
            )
        else:
            selected = np.nonzero(
                (
                    (self._decs >= self._dec_low)
                    & (self._decs <= self._dec_high)
                    & (self._ras >= self._ra_low)
                    & (self._ras <= self._ra_high)
                )
            )
        selected_dec_band = np.nonzero(
            ((self._decs >= self._dec_low) & (self._decs <= self._dec_high))
        )

        self._selected = selected

        self._selected_ras = self._ras[selected]

        self._selected_decs = self._decs[selected]

        self._selected_energies = self._energies[selected]

        self._selected_bg_energies = self._energies  # [selected_dec_band]

        self._selected_bg_ras = self._ras  # [selected_dec_band]

        self._selected_bg_decs = self._decs  # [selected_dec_band]

        if isinstance(self._ang_errs, np.ndarray):
            self._selected_ang_errs = self._ang_errs[selected]
        else:
            self._selected_ang_errs = [1] * len(selected[0])

        self.Nprime = len(selected[0])

        self.N = self._energies.size  # len(selected_dec_band[0])

        if isinstance(
            self._direction_likelihood, EventDependentSpatialGaussianLikelihood
        ) or isinstance(self._direction_likelihood, RayleighDistribution):
            self._signal_llh_spatial = self._direction_likelihood(
                self._selected_ang_errs,
                self._selected_ras,
                self._selected_decs,
                self._source_coord,
            )

        if isinstance(self._bg_energy_likelihood, DataDrivenBackgroundLikelihood):
            self._bg_llh = self._bg_energy_likelihood(
                self._selected_energies, 2.0, self._selected_decs
            )
            self._bg_llh[np.nonzero(self._bg_llh == 0.0)] = 1e-10

    def _signal_likelihood(
        self,
        ra: np.ndarray,
        dec: np.ndarray,
        source_coord: Tuple[float, float],
        energy: np.ndarray,
        index: float,
        ang_err: np.ndarray,
    ):
        """
        Calculate the signal likelihood of a given event.
        :param ra: RA of event, in rad
        :param dec: DEC of event, in rad
        :param source_coord: Tuple of source coordinate (ra, dec), in rad
        :param energy: Energy of event in GeV
        :param index: Spectral index of source model
        :param ang_err: Angular error on the event, in degrees
        :return: Likelihood for each provided event
        """

        if isinstance(
            self._direction_likelihood, EnergyDependentSpatialGaussianLikelihood
        ):

            def spatial():
                return self._direction_likelihood(ra, dec, source_coord, energy, index)

            def en():
                return self._energy_likelihood(energy, index, dec)

        elif isinstance(
            self._direction_likelihood, EventDependentSpatialGaussianLikelihood
        ) or isinstance(self._direction_likelihood, RayleighDistribution):

            def spatial():
                return self._signal_llh_spatial

            def en():
                return self._energy_likelihood(energy, index, dec)

        else:

            def spatial():
                return self._direction_likelihood(ra, dec, source_coord)

            def en():
                return self._energy_likelihood(energy, index, dec)

        if self.which == "spatial":
            output = spatial()
        elif self.which == "energy":
            output = en()
        else:
            output = en() * spatial()

        return output

    def _background_likelihood(
        self,
        energy: np.ndarray,
        dec: np.ndarray,
        weight: float = 0.0,
        index_astro: float = 2.5,
        index_atmo: float = 3.7,
    ):
        """
        Calculate the background likelihood for an event of given energy.
        Split this into two background contributions:
        one for atmospheric events (atmo_index) and one for astrophysical events (astro_index)
        Likelihood is sum of two components, `weight` is parameter of simplex,
        weight = 0 -> fully atmospheric, weight = 1 -> fully astrophysical
        :param energy: Energy of events in GeV
        :param dec: Declination of events in rad
        :param weight: Weight of backgrounds, 0 for pure atmospheric (index=3.7), 1 for pure astrophysical (index=2.5)
        :param index_astro: Astrophysical background spectral index, defaults to 2.5
        :param index_atmo: Atmospheric background spectral index, defaults to 3.7
        :return: Likelihood for each provided event
        """

        # Check for background being completely determined by data likelihood
        if isinstance(
            self._bg_energy_likelihood, DataDrivenBackgroundLikelihood
        ) and isinstance(self._bg_spatial_likelihood, DataDrivenBackgroundLikelihood):
            output = self._bg_llh
            # output[np.nonzero(output==0.)] = 1e-10
            return output

        if self._bg_energy_likelihood is not None:
            if isinstance(
                self._bg_energy_likelihood, DataDrivenBackgroundEnergyLikelihood
            ):

                def en(energy, index, dec):
                    return self._bg_energy_likelihood(energy, index, dec)

            else:

                def en(energy):
                    return self._bg_energy_likelihood(energy)

        else:

            def en(energy, index, dec):
                return self._energy_likelihood(energy, index, dec)

        if self._bg_spatial_likelihood is not None:

            def spatial(dec):
                return self._bg_spatial_likelihood(dec)

        else:

            def spatial(dec):
                return np.full(energy.shape, 1.0 / self._band_solid_angle)

        # Check which part is used for likelihood calculation
        if self.which == "spatial":
            output = spatial(dec)
        else:
            if np.isclose(weight, 0):
                if self.which == "energy":
                    output = en(energy, index_atmo, dec)
                else:
                    output = en(energy, index_atmo, dec) * spatial(dec)
            else:
                if self.which == "energy":
                    output = (1 - weight) * en(energy, index_atmo, dec) + weight * en(
                        energy, index_astro, dec
                    )
                else:
                    output = (
                        (1 - weight) * en(energy, index_atmo, dec)
                        + weight * en(energy, index_astro, dec)
                    ) * spatial(dec)

        output[np.nonzero(output == 0)] = 1e-10

        return output

    def _func_to_minimize(
        self,
        ns: float,
        index: float = 2.0,
        weight: float = 0.0,
        index_astro: float = 2.5,
        index_atmo: float = 3.7,
    ):
        """
        Calculate the -log(likelihood_ratio) for minimization.

        Uses calculation described in:
        https://github.com/IceCubeOpenSource/SkyLLH/blob/master/doc/user_manual.pdf

        If there is a prior, it is added here, as this is equivalent to maximising
        the likelihood.

        :param ns: Number of source counts
        :param index: Spectral index of the source, defaults to 2.0
        :param weight: Weight of backgrounds, 0 for pure atmospheric (index=3.7), 1 for pure astrophysical (index=2.5)
        :param index_astro: Astrophysical background spectral index, defaults to 2.5
        :param index_atmo: Atmospheric background spectral index, defaults to 3.7
        :return: negative log likelihood ratio over all events
        """

        one_plus_alpha = 1e-10
        alpha = one_plus_alpha - 1
        if isinstance(self._energy_likelihood, MarginalisedIntegratedEnergyLikelihood):
            index_list = [index]
        else:
            idx = np.digitize(index, self._energy_likelihood.index_list)
            llhs = np.zeros(2)
            index_list = self._energy_likelihood.index_list[idx - 1 : idx + 1]

        for c, indx in enumerate(index_list):
            log_likelihood_ratio = np.zeros_like(self._selected_ras)
            signal = self._signal_likelihood(
                self._selected_ras,
                self._selected_decs,
                self._source_coord,
                self._selected_energies,
                indx,
                ang_err=self._selected_ang_errs,
            )

            bg = self._background_likelihood(
                self._selected_energies,
                self._selected_decs,
                weight,
                index_astro,
                index_atmo,
            )

            chi = (1 / self.N) * (signal / bg - 1)

            alpha_i = ns * chi

            one_p = 1 + alpha_i < one_plus_alpha

            alpha_tilde = (alpha_i[one_p] - alpha) / one_plus_alpha
            log_likelihood_ratio[one_p] = (
                np.log1p(alpha) + alpha_tilde - 0.5 * np.power(alpha_tilde, 2)
            )
            log_likelihood_ratio[~one_p] = np.log1p(alpha_i[~one_p])
            log_likelihood_ratio = np.sum(log_likelihood_ratio)

            log_likelihood_ratio += (self.N - self.Nprime) * np.log1p(-ns / self.N)

            if isinstance(
                self._energy_likelihood, MarginalisedIntegratedEnergyLikelihood
            ):
                # exit for integrated likelihood after one iteration, since that's all that's needed
                if self._index_prior:
                    log_likelihood_ratio += np.log(self._index_prior(indx))
                return -log_likelihood_ratio
            else:
                # continue through loop, pick value in the middle at the end
                llhs[c] = log_likelihood_ratio

        log_likelihood_ratio = np.interp(index, index_list, llhs)
        if self._index_prior:
            log_likelihood_ratio += np.log(self._index_prior(indx))

        return -log_likelihood_ratio

    def _func_to_minimize_sp(self, ns: float, index: float = 2.0):
        """
        Calculate the -log(likelihood_ratio) for minimization using spatial information only.

        Uses calculation described in:
        https://github.com/IceCubeOpenSource/SkyLLH/blob/master/doc/user_manual.pdf

        If there is a prior, it is added here, as this is equivalent to maximising
        the likelihood.

        :param ns: Number of source counts.
        :param index: Dummy argument
        """

        one_plus_alpha = 1e-10
        alpha = one_plus_alpha - 1

        log_likelihood_ratio = np.zeros_like(self._selected_ras)

        signal = self._signal_likelihood(
            self._selected_ras,
            self._selected_decs,
            self._source_coord,
            self._selected_energies,
            2.0,
            ang_err=self._selected_ang_errs,
        )

        bg = self._background_likelihood(self._selected_energies, self._selected_decs)

        chi = (1 / self.N) * (signal / bg - 1)

        alpha_i = ns * chi

        one_p = 1 + alpha_i < one_plus_alpha

        alpha_tilde = (alpha_i[one_p] - alpha) / one_plus_alpha
        log_likelihood_ratio[one_p] = (
            np.log1p(alpha) + alpha_tilde - 0.5 * np.power(alpha_tilde, 2)
        )
        log_likelihood_ratio[~one_p] = np.log1p(alpha_i[~one_p])
        log_likelihood_ratio = np.sum(log_likelihood_ratio)

        log_likelihood_ratio += (self.N - self.Nprime) * np.log1p(-ns / self.N)

        return -log_likelihood_ratio

    def _func_to_minimize_bg(self, weight=0.0, index_astro=2.5, index_atmo=3.7):
        """
        Negative loglike of background only
        :param weight: Weight for background components, 0 -> fully atmospheric, 1 -> fully astrophysical
        :param index_astro: Astrophysical spectral index
        :param index_atmo: Atmospherical spectral index
        :return: Negative loglikelihood
        """

        likelihood = self._background_likelihood(
            self._selected_bg_energies,
            self._selected_bg_decs,
            weight=weight,
            index_atmo=index_atmo,
            index_astro=index_astro,
        )
        return -np.sum(np.log(likelihood))

    def __call__(self, ns, index, weight=0, index_astro=2.5, index_atmo=3.7):
        """
        Wrapper function for convenience.
        """

        return self._func_to_minimize(ns, index, weight, index_astro, index_atmo)

    def _minimize(self):
        """
        Minimize -log(likelihood_ratio) for the source hypothesis,
        returning the best fit ns and index.

        Uses the iMiuint wrapper.
        """

        init_index = (
            self._energy_likelihood._min_index
            + (self._max_index - self._energy_likelihood._min_index) / 2
        )
        init_ns = self._ns_min + (self._ns_max - self._ns_min) / 2
        init_weight = 0.0
        init_astro = 2.5
        init_atmo = 3.7

        if self.which == "spatial":
            # Spatial-only likelihood needs special function, because no spectral index is used
            logger.debug("Using only spatial information.")
            self.minimize_this = self._func_to_minimize_sp
        else:
            self.minimize_this = self._func_to_minimize
            logger.debug("Using all information.")

        if self.which != "spatial":
            m = Minuit(
                self.minimize_this,
                ns=init_ns,
                index=init_index,
                weight=init_weight,
                index_astro=init_astro,
                index_atmo=init_atmo,
            )
        else:
            m = Minuit(
                self.minimize_this,
                ns=init_ns,
                index=init_index,
            )

        m.limits["ns"] = (self._ns_min, self._ns_max)
        m.limits["index"] = (
            self._energy_likelihood._min_index,
            self._energy_likelihood._max_index,
        )
        m.errors["ns"] = 1
        m.errors["index"] = 0.1

        if self.which != "spatial":
            m.limits["weight"] = (0.0, 1.0)
            m.limits["index_astro"] = (
                self._energy_likelihood._min_index,
                self._energy_likelihood._max_index,
            )
            m.limits["index_atmo"] = (
                self._energy_likelihood._min_index,
                self._energy_likelihood._max_index,
            )
            m.errors["weight"] = 0.05
            m.errors["index_atmo"] = 0.1
            m.errors["index_astro"] = 0.1
            if ~self._vary_atmo:
                m.fixed["index_atmo"] = True
            if not (self._vary_atmo and self._vary_astro):
                # only let astro vary, if atmo is also varied, else atmo is only background
                m.fixed["weight"] = True
                m.fixed["index_astro"] = True

        elif self.which == "spatial":
            m.fixed["index"] = True
            # m.fixed["index_atmo"] = True
            # m.fixed["index_astro"] = True
            # m.fixed["weight"] = True

        m.errordef = 0.5
        m.migrad()

        if not m.valid or not m.fmin.has_accurate_covar or not m.fmin.has_covariance:

            # Fix the index as can be uninformative
            m.fixed["index"] = True
            m.fixed["index_atmo"] = True
            m.fixed["index_astro"] = True
            m.migrad()

        self._best_fit_ns = m.values["ns"]
        self._best_fit_index = m.values["index"]
        self.m = m
        return m

    def _minimize_bg(self, astro: bool = False):
        """
        Minimize the background negative log-ikelihood only.
        """

        init_astro = 2.5
        init_atmo = 3.3
        if astro:
            init_weight = 0.2
        else:
            init_weight = 0.0

        m = Minuit(
            self._func_to_minimize_bg,
            weight=init_weight,
            index_astro=init_astro,
            index_atmo=init_atmo,
        )
        m.errordef = 0.5
        m.errors["index_astro"] = 0.1
        m.errors["index_atmo"] = 0.1
        m.limits["index_astro"] = (self._min_index, self._max_index)
        m.limits["index_atmo"] = (self._min_index, self._max_index)
        m.limits["weight"] = (0, 1)
        if not astro:
            m.fixed["index_astro"] = True
            m.fixed["weight"] = True

        m.migrad()

        self.m = m
        return m

    def _minimize_grid(self):
        """
        Minimize -log(likelihood_ratio) for the source hypothesis,
        returning the best fit ns and index.

        This simple grid method takes roughly the same time as minuit
        and is more accurate...
        """

        ns_grid = np.linspace(self._ns_min, self._ns_max, 10)
        index_grid = np.linspace(
            self._energy_likelihood._min_index, self._max_index, 10
        )

        out = np.zeros((len(ns_grid), len(index_grid)))
        for i, ns in enumerate(ns_grid):
            for j, index in enumerate(index_grid):
                out[i][j] = self._func_to_minimize(ns, index)

        sel = np.where(out == np.min(out))

        if len(sel[0]) > 1:

            self._best_fit_index = 3.7
            self._best_fit_ns = 0.0

        else:

            self._best_fit_ns = ns_grid[sel[0]][0]
            self._best_fit_index = index_grid[sel[1]][0]
        self.grid = out

    def _first_derivative_likelihood_ratio(self, ns=0, index=2.0):
        """
        First derivative of the likelihood ratio.
        Equation 41 in
        https://github.com/IceCubeOpenSource/SkyLLH/blob/master/doc/user_manual.pdf.
        """

        one_plus_alpha = 1e-10
        alpha = one_plus_alpha - 1

        self._first_derivative = []

        for i in range(self.Nprime):

            signal = self._signal_likelihood(
                self._selected_ras[i],
                self._selected_decs[i],
                self._source_coord,
                self._selected_energies[i],
                index,
                ang_err=self._selected_ang_errs[i],
            )

            bg = self._background_likelihood(self._selected_energies[i])

            chi_i = (1 / self.N) * ((signal / bg) - 1)

            alpha_i = ns * chi_i

            if (1 + alpha_i) < one_plus_alpha:

                alpha_tilde = (alpha_i - alpha) / one_plus_alpha

                self._first_derivative.append(
                    (1 / one_plus_alpha) * (1 - alpha_tilde) * chi_i
                )

            else:

                self._first_derivative.append(chi_i / (1 + alpha_i))

        self._first_derivative = np.array(self._first_derivative)

        return sum(self._first_derivative) - ((self.N - self.Nprime) / (self.N - ns))

    def _second_derivative_likelihood_ratio(self, ns=0):
        """
        Second derivative of the likelihood ratio.
        Equation 44 in
        https://github.com/IceCubeOpenSource/SkyLLH/blob/master/doc/user_manual.pdf.
        """

        self._second_derivative = -((self._first_derivative) ** 2)

        return sum(self._second_derivative) - (
            (self.N - self.Nprime) / (self.N - ns) ** 2
        )

    def get_test_statistic(self):
        """
        Calculate the test statistic for the best fit ns
        """

        self._m = self._minimize()
        # self._minimize_grid()

        # For resolving the TS peak at zero
        # if self._best_fit_ns == 0:

        #    first_der = self._first_derivative_likelihood_ratio(self._best_fit_ns, self._best_fit_index)
        #    second_der = self._second_derivative_likelihood_ratio(self._best_fit_ns)

        #    self.test_statistic = -2 * (first_der**2 / (4 * second_der))

        #   self.likelihood_ratio = np.exp(-self.test_statistic/2)

        # else:

        # make sure that correct minimum (i.e. spatial/energy/both) is calculated:
        if self.which != "spatial":
            neg_log_lik = self.minimize_this(
                self._best_fit_ns,
                self._best_fit_index,
                self.m.values["weight"],
                self.m.values["index_astro"],
                self.m.values["index_atmo"],
            )
        else:
            neg_log_lik = self.minimize_this(self._best_fit_ns, self._best_fit_index)

        self.likelihood_ratio = np.exp(neg_log_lik)

        self.test_statistic = -2 * neg_log_lik

        return self.test_statistic


class SpatialOnlyPointSourceLikelihood:
    """
    Calculate the point source likelihood for a given
    neutrino dataset - in terms of reconstructed
    arrival directions.

    This class is exactly as in PointSourceLikelihood,
    but without the energy depedence.

    Should be removed at some point, this case is already
    included in the main class with the keyword "which",
    defaulting to "both".
    """

    def __init__(self, direction_likelihood, event_coords, source_coord):
        """
        Calculate the point source likelihood for a given
        neutrino dataset - in terms of reconstructed
        energies and arrival directions.

        :param direction_likelihood: An instance of SpatialGaussianLikelihood.
        :param event_coords: List of (ra, dec) tuples for reconstructed coords.
        :param source_coord: (ra, dec) pf the point to test.
        """

        self._direction_likelihood = direction_likelihood

        self._band_width = 3 * self._direction_likelihood._sigma  # degrees

        dec_low = source_coord[1] - np.deg2rad(self._band_width)
        dec_high = source_coord[1] + np.deg2rad(self._band_width)
        self._band_solid_angle = 2 * np.pi * (np.sin(dec_high) - np.sin(dec_low))

        self._event_coords = event_coords

        self._source_coord = source_coord

        self._bg_index = 3.7

        self._ns_min = 0.0
        self._ns_max = 100
        self._max_index = 3.7

        self._select_nearby_events()

        self.Ntot = len(self._event_coords)

    def _select_nearby_events(self):

        ras = np.array([_[0] for _ in self._event_coords])

        decs = np.array([_[1] for _ in self._event_coords])

        source_ra, source_dec = self._source_coord

        dec_fac = np.deg2rad(self._band_width)

        selected = list(
            set(
                np.where(
                    (decs >= source_dec - dec_fac)
                    & (decs <= source_dec + dec_fac)
                    & (ras >= source_ra - dec_fac)
                    & (ras <= source_ra + dec_fac)
                )[0]
            )
        )

        selected_dec_band = np.where(
            (decs >= source_dec - dec_fac) & (decs <= source_dec + dec_fac)
        )[0]

        self._selected = selected

        self._selected_event_coords = [
            (ec[0], ec[1])
            for ec in self._event_coords
            if (ec[1] >= source_dec - dec_fac)
            & (ec[1] <= source_dec + dec_fac)
            & (ec[0] >= source_ra - dec_fac)
            & (ec[0] <= source_ra + dec_fac)
        ]

        self.Nprime = len(selected)

        self.N = len(selected_dec_band)

    def _signal_likelihood(self, event_coord, source_coord):

        return self._direction_likelihood(event_coord, source_coord)

    def _background_likelihood(self):

        return 1.0 / self._band_solid_angle

    def _get_neg_log_likelihood_ratio(self, ns):
        """
        Calculate the -log(likelihood_ratio) for minimization.

        Uses calculation described in:
        https://github.com/IceCubeOpenSource/SkyLLH/blob/master/doc/user_manual.pdf

        :param ns: Number of source counts.
        """

        one_plus_alpha = 1e-10
        alpha = one_plus_alpha - 1

        log_likelihood_ratio = 0.0

        for i in range(self.Nprime):

            signal = self._signal_likelihood(
                self._selected_event_coords[i], self._source_coord
            )

            bg = self._background_likelihood()

            chi = (1 / self.N) * (signal / bg - 1)

            alpha_i = ns * chi

            if (1 + alpha_i) < one_plus_alpha:

                alpha_tilde = (alpha_i - alpha) / one_plus_alpha
                log_likelihood_ratio += (
                    np.log1p(alpha) + alpha_tilde - (0.5 * alpha_tilde**2)
                )

            else:

                log_likelihood_ratio += np.log1p(alpha_i)

        log_likelihood_ratio += (self.N - self.Nprime) * np.log1p(-ns / self.N)

        return -log_likelihood_ratio

    def __call__(self, ns):
        """
        Wrapper function for convenience.
        """

        return self._get_neg_log_likelihood_ratio(ns)

    def _minimize(self):
        """
        Minimize -log(likelihood_ratio) for the source hypothesis,
        returning the best fit ns and index.

        Uses the iMiuint wrapper.
        """

        init_ns = self._ns_min + (self._ns_max - self._ns_min) / 2

        m = Minuit(
            self._func_to_minimize,
            ns=init_ns,
        )
        m.limits["ns"] = (self._ns_min, self._ns_max)
        m.errors["ns"] = 1
        m.errordef = 0.5
        m.migrad()

        self._best_fit_ns = m.values["ns"]

        return m

    def get_test_statistic(self):
        """
        Calculate the test statistic for the best fit ns
        """

        self._minimize()

        neg_log_lik = self._get_neg_log_likelihood_ratio(self._best_fit_ns)

        self.likelihood_ratio = np.exp(neg_log_lik)

        self.test_statistic = -2 * neg_log_lik

        return self.test_statistic


class TimeDependentPointSourceLikelihood:
    def __init__(
        self,
        source_coord: Tuple[float, float],
        data_periods: List[str],
        ra: Dict,
        dec: Dict,
        reco_energy: Dict,
        ang_err: Dict,
        energy_llh: Dict = None,
        times: Dict = None,
        path=None,
        index_list=None,
        vary_atmo: bool = False,
        vary_astro: bool = False,
        which: str = "both",
        emin: float = 1e1,
        emax: float = 1e9,
        min_index: float = 1.5,
        max_index: float = 5.0,
        new_reco_bins: np.ndarray = np.linspace(1, 9, num=25),
        sigma: float = 2.0,
        band_width_factor: float = 5.0,
    ):
        """
        Create likelihood covering multiple data taking periods.
        :param source_coord: Tuple of ra, dec, in rad
        :param periods: List of str of period names, eg. `IC40`. Only periods with IRF, e.g. no IC86_III !!!
        :param ra: Dict of RAs
        :param dec: Dict of DECs
        :param reco_energy: Dict of reconstructed energies in GeV
        :param ang_err: Dict of 68% angular errors in degrees
        :param energy_llh: Dict of objects inheriting from MarginalisedEnergyLikelihood
        :param times: Dict of observational times in years (without astropy.unit attached)
        :param path: Path to simualation files if energy likelihood is based on simulations
        :param index_list: List of indices covered by the events used to build the energy likelihood.
        :param which: String, `both`, `spatial`, `energy` indicating which likelihoods are to be used.
        :param emin: Minimum reco energy considered
        :param emax: Maximum reco energy considered
        :param min_index: Minimum spectral index
        :param max_index: Maximum spectral index
        :param new_reco_bins: Reco energy bins at which energy likelihood is evaluated
        :param sigma: Worst angular resolution considered for spatial likelihood, defaults to 5 degrees
        :param band_width_factor: Factor multiplied with sigma for event selection, defaults to 3
        """

        if which not in ["both", "energy", "spatial"]:
            raise ValueError("Provided likelihood type not provided")

        self.which = which
        # do not call setter here, needed attributes do not exist yet
        self._source_coord = source_coord
        self._data_periods = data_periods
        self._uptime = Uptime(*data_periods)
        self._irf_periods = self._uptime.irf_periods
        self.index_list = index_list
        self._min_index = min_index
        self._max_index = max_index
        self._vary_atmo = vary_atmo
        self._vary_astro = vary_astro
        self.likelihoods = OrderedDict()
        # Can use one spatial llh for all periods, 'tis but a Gaussian
        spatial_llh = RayleighDistribution(sigma=sigma)
        if times is None:
            self.times = self._uptime.cumulative_time_obs()
        else:
            self.times = times
        # self.tirf = TimeDependentIceCube.from_periods(*self._irf_periods)
        self.aeffs = [
            EffectiveArea.from_dataset("20210126", p, spline_degree=3)
            for p in self._irf_periods
        ]
        self.nu_calcs = {}
        self.flux = PowerLawFlux(1e-20, 1e5, 2.5, lower_energy=emin, upper_energy=emax)
        self.source = PointSource(flux_model=self.flux, z=0.0, coord=self.source_coord)

        if energy_llh is None:
            energy_llh = {}
            create_e_llh = True
        else:
            create_e_llh = False

        for c, p in enumerate(self._irf_periods):
            if create_e_llh:
                energy_llh[p] = MarginalisedIntegratedEnergyLikelihood(
                    p, new_reco_bins, self._min_index, self._max_index
                )

            self.nu_calcs[p] = NeutrinoCalculator(
                [self.source],
                self.aeffs[c],
                # energy_resolution=energy_llh[p] if create_e_llh else None
            )
            # create likelihood objects

            # bg_llh = DataDrivenBackgroundLikelihood(p, spline=0)

            self.likelihoods[p] = PointSourceLikelihood(
                spatial_llh,
                energy_llh[p],
                ra[p],
                dec[p],
                reco_energy[p],
                ang_err[p],
                self.source_coord,
                which=self.which,
                band_width_factor=band_width_factor,
                bg_energy_likelihood=DataDrivenBackgroundEnergyLikelihood(period=p),
                bg_spatial_likelihood=DataDrivenBackgroundSpatialLikelihood(period=p),
                # bg_energy_likelihood=bg_llh,
                # bg_spatial_likelihood=bg_llh,
            )

    @property
    def source_coord(self):
        return self._source_coord

    @source_coord.setter
    def source_coord(self, new_coord):
        self._source_coord = new_coord
        # update nutrino calculators:
        self.source = PointSource(flux_model=self.flux, z=0.0, coord=new_coord)
        for p in self._irf_periods:
            self.nu_calcs[p]._sources = [self.source]
        # update likelihoods
        for p in self._irf_periods:
            self.likelihoods[p].source_coord = (
                new_coord  # calls setter for single-seasons's likelihood
            )

    def reset_events(self, ra: Dict, dec: Dict, reco_energy: Dict, ang_err: Dict):
        logger.info("Resetting events.")
        for p in self._irf_periods:
            self.likelihoods[p].update_events(ra[p], dec[p], reco_energy[p], ang_err[p])

    def __call__(self, ns: float, index: float):
        """
        Calculate negative log-like ratio as function of ns and index.
        ns is now vector with an entry for each period.
        :param ns: Number of source events.
        :param index: Spectral index of source spectrum.
        :return: Negative log likelihood ratio
        """

        return self._func_to_minimize(ns, index)

    def _func_to_minimize(self, ns, index, weight=0.0, index_astro=2.5, index_atmo=3.7):
        """
        According to https://github.com/icecube/skyllh/blob/master/doc/user_manual.pdf,
        Eq. (59), the returned values of each period's llh._func_to_minimize() can be added.
        :param ns: Number of source events
        :param index: Spectral index of source spectrum
        :return: negative log likelihood ratio
        """
        neg_log_like = 0
        weights = self._calc_weights(index)
        for w, p in zip(weights, self._irf_periods):
            if (
                self.likelihoods[p].N == 0
            ):  # or np.isclose(ns * w / llh.N - 1., 0., atol=1e-10):
                # is this appropriate?
                continue
            val = self.likelihoods[p](ns * w, index)
            neg_log_like += val
        return neg_log_like

    def _func_to_minimize_bg(self, weight=0.0, index_astro=2.5, index_atmo=3.7):
        """
        Negative loglike of background only
        :param weight: Weight for background components, 0 -> fully atmospheric, 1 -> fully astrophysical
        :param index_astro: Astrophysical spectral index
        :param index_atmo: Atmospherical spectral index
        :return: Negative loglikelihood
        """

        neg_log_like = 0
        for llh in self.likelihoods.values():
            likelihood = llh._func_to_minimize_bg(
                weight=weight, index_astro=index_astro, index_atmo=index_atmo
            )
            neg_log_like += likelihood
        return neg_log_like

    def _func_to_minimize_sp(
        self, ns, index, weight=0.0, index_astro=2.5, index_atmo=3.7
    ):
        """
        According to https://github.com/icecube/skyllh/blob/master/doc/user_manual.pdf,
        Eq. (59), the returned values of each period's llh._func_to_minimize() can be added.
        :param arg: numpy.ndarray, last entry is index, all before are number of source events.
        """
        neg_log_like = 0
        weights = [1]
        for w, llh in zip(weights, self.likelihoods.values()):
            val = llh._func_to_minimize_sp(ns)
            neg_log_like += val
        return neg_log_like

    def _minimize(self):
        """
        Minimize -log(likelihood_ratio) for the source hypothesis,
        returning the best fit ns and index.

        Uses the iMinuint wrapper.
        """

        some_llh = self.likelihoods[list(self.likelihoods.keys())[0]]
        init_index = (
            some_llh._min_index + (some_llh._max_index - some_llh._min_index) / 2
        )
        limit_index = (
            some_llh._energy_likelihood._min_index,
            some_llh._energy_likelihood._max_index,
        )
        ns_max = []
        for llh in self.likelihoods.values():
            ns_max.append(llh._ns_max)
        # init_ns = min(ns_max) * 0.01
        init_ns = 2.0
        init = [init_ns, init_index, 0, 2.5, 3.7]

        # for limit ns:
        # find all products of weight * ns and see where it will crash first
        smallest_N = min([llh.N for llh in self.likelihoods.values()])
        limits = [(0, smallest_N), limit_index, (0, 1), limit_index, limit_index]

        # Get errors to start with
        errors = [1, 0.1, 0.1, 0.1, 0.1]
        name = ["ns", "index", "weight", "index_astro", "index_atmo"]

        if self.which == "spatial":
            # raise ValueError("Currently not supported")
            # Only spatial-only likelihood needs special function, because no spectral index is used
            self.minimize_this = self._func_to_minimize_sp
        else:
            self.minimize_this = self._func_to_minimize
            # add errors, limits, start value and name of index

        self.m = Minuit(self.minimize_this, *init, name=name)
        self.m.errordef = 0.5
        self.m.errors = errors
        self.m.limits = limits
        if not self._vary_atmo:
            self.m.fixed["index_atmo"] = True
            self.m.fixed["index_astro"] = True
            self.m.fixed["weight"] = True
        elif not self._vary_astro:
            self.m.fixed["index_astro"] = True
            self.m.fixed["weight"] = True
        if self.which == "spatial":
            self.m.fixed["index"] = True

        self.m.migrad()
        # self.m.scipy("SLSQP")

        if self.which != "spatial":
            if not self.m.fmin.is_valid or not self.m.fmin.has_covariance:

                # Fix the index as can be uninformative
                self.m.fixed["index"] = True
                self.m.fixed["index_atmo"] = True
                self.m.fixed["index_astro"] = True
                self.m.fixed["weight"] = True
                self.m.migrad()
        else:
            if not self.m.fmin.is_valid or not self.m.fmin.has_covariance:
                logger.warning("Fit has not converged, proceed with caution.")

        if self.which != "spatial":
            self._best_fit_index = self.m.values["index"]
        else:
            self._best_fit_index = 2.7
        self._best_fit_ns = self.m.values["ns"]
        return self.m

    def _minimize_bg(self, astro: bool = False):
        """
        Minimize the background negative log-ikelihood only.
        """

        init_astro = 2.5
        init_atmo = 3.3
        if astro:
            init_weight = 0.2
        else:
            init_weight = 0.0

        m = Minuit(
            self._func_to_minimize_bg,
            weight=init_weight,
            index_astro=init_astro,
            index_atmo=init_atmo,
        )
        m.errordef = 0.5
        m.errors["index_astro"] = 0.1
        m.errors["index_atmo"] = 0.1
        m.limits["index_astro"] = (self._min_index, self._max_index)
        m.limits["index_atmo"] = (self._min_index, self._max_index)
        m.limits["weight"] = (0, 1)
        if not astro:
            m.fixed["index_astro"] = True
            m.fixed["weight"] = True

        m.migrad()

        self.m = m
        return m

    def _calc_weights(self, index: float):
        """
        Calculate weights (i.e. number of expected events) for the individual likelihoods'
        :param index: Spectral index
        """
        # works as intended, same numbers as NeutrinoCalculator in simulate.md example
        # TODO write test?
        n_i = np.zeros(len(self._irf_periods))
        if self.which != "spatial":
            for c, p in enumerate(self._irf_periods):
                self.nu_calcs[p]._sources[0]._flux_model._index = index
                n_i[c] = self.nu_calcs[p](
                    time=self.times[p],
                )[0]
        else:
            for c, p in enumerate(self._irf_periods):
                n_i[c] = self.times[p]
        N = np.sum(n_i)
        weights = n_i / N

        return weights

    def ns_to_flux(self, ns: float, index: float):
        """
        Convert some given ns and spectral index to the average flux
        over the detector livetime.
        """

        raise NotImplementedError

    def _update_flux(self, flux):

        raise NotImplementedError

    def get_test_statistic(self):
        """
        Calculate test statistic
        """

        self._minimize()
        neg_log_lik = self.minimize_this(self._best_fit_ns, self._best_fit_index)
        self.likelihood_ratio = np.exp(neg_log_lik)
        self.test_statistic = -2 * neg_log_lik
        return self.test_statistic

    @property
    def Ntot(self):
        n = 0
        for l in self.likelihoods.values():
            n += l.Ntot
        return n

    @property
    def Ntot_dict(self):
        n = {}
        for c, v in self.likelihoods.items():
            n[c] = v.Ntot
        return n

    @property
    def N(self):
        n = 0
        for l in self.likelihoods.values():
            n += l.N
        return n

    @property
    def N_dict(self):
        n = {}
        for c, v in self.likelihoods.items():
            n[c] = v.N
        return n

    @property
    def Nprime(self):
        n = 0
        for l in self.likelihoods.values():
            n += l.Nprime
        return n

    @property
    def Nprime_dict(self):
        n = {}
        for c, v in self.likelihoods.items():
            n[c] = v.Nprime
        return n


class EnergyDependentSpatialPointSourceLikelihood:
    """
    Calculate the point source likelihood for a given
    neutrino dataset - in terms of reconstructed
    arrival directions.

    This class is exactly as in PointSourceLikelihood,
    but without the energy depedence.
    """

    def __init__(
        self,
        direction_likelihood,
        ras,
        decs,
        energies,
        source_coord,
        band_width_factor=3.0,
    ):
        """
        Calculate the point source likelihood for a given
        neutrino dataset - in terms of reconstructed
        energies and arrival directions.

        :param direction_likelihood: An instance of SpatialGaussianLikelihood.
        :param ras: Array of right acsensions in [rad]
        :param decs: Array of declinations in [dec]
        :param source_coord: (ra, dec) pf the point to test.
        """

        self._direction_likelihood = direction_likelihood

        self._band_width = band_width_factor * self._direction_likelihood.get_low_res()

        self._dec_low = source_coord[1] - np.deg2rad(self._band_width)

        self._dec_high = source_coord[1] + np.deg2rad(self._band_width)

        if self._dec_low < np.arcsin(-0.1) or np.isnan(self._dec_low):
            self._dec_low = np.arcsin(-0.1)

        if self._dec_high > np.arcsin(1.0) or np.isnan(self._dec_high):
            self._dec_high = np.arcsin(1.0)

        self._band_solid_angle = (
            2 * np.pi * (np.sin(self._dec_high) - np.sin(self._dec_low))
        )

        self._ra_low = source_coord[0] - np.deg2rad(self._band_width)

        self._ra_high = source_coord[0] + np.deg2rad(self._band_width)

        self._ras = ras

        self._decs = decs

        self._source_coord = source_coord

        self._energies = energies

        self._ns_min = 0.0
        self._ns_max = 100

        self._select_nearby_events()

        self.Ntot = len(self._ras)

    def _select_nearby_events(self):

        source_ra, source_dec = self._source_coord

        selected = list(
            set(
                np.where(
                    (self._decs >= self._dec_low)
                    & (self._decs <= self._dec_high)
                    & (self._ras >= self._ra_low)
                    & (self._ras <= self._ra_high)
                )[0]
            )
        )

        selected_dec_band = np.where(
            (self._decs >= self._dec_low) & (self._decs <= self._dec_high)
        )[0]

        self._selected = selected

        self._selected_ras = self._ras[selected]

        self._selected_decs = self._decs[selected]

        self._selected_energies = self._energies[selected]

        self.Nprime = len(selected)

        self.N = len(selected_dec_band)

    def _signal_likelihood(self, ra, dec, source_coord, energy):

        return self._direction_likelihood((ra, dec), source_coord, energy)

    def _background_likelihood(self):

        return 1.0 / self._band_solid_angle

    def _get_neg_log_likelihood_ratio(self, ns):
        """
        Calculate the -log(likelihood_ratio) for minimization.

        Uses calculation described in:
        https://github.com/IceCubeOpenSource/SkyLLH/blob/master/doc/user_manual.pdf

        :param ns: Number of source counts.
        """

        one_plus_alpha = 1e-10
        alpha = one_plus_alpha - 1

        log_likelihood_ratio = 0.0

        for i in range(self.Nprime):

            signal = self._signal_likelihood(
                self._selected_ras[i],
                self._selected_decs[i],
                self._source_coord,
                self._energies[i],
            )

            bg = self._background_likelihood()

            chi = (1 / self.N) * (signal / bg - 1)

            alpha_i = ns * chi

            if (1 + alpha_i) < one_plus_alpha:

                alpha_tilde = (alpha_i - alpha) / one_plus_alpha
                log_likelihood_ratio += (
                    np.log1p(alpha) + alpha_tilde - (0.5 * alpha_tilde**2)
                )

            else:

                log_likelihood_ratio += np.log1p(alpha_i)

        log_likelihood_ratio += (self.N - self.Nprime) * np.log1p(-ns / self.N)

        return -log_likelihood_ratio

    def __call__(self, ns):
        """
        Wrapper function for convenience.
        """

        return self._get_neg_log_likelihood_ratio(ns)

    def _minimize(self):
        """
        Minimize -log(likelihood_ratio) for the source hypothesis,
        returning the best fit ns and index.

        Uses the iMiuint wrapper.
        """

        init_ns = self._ns_min + (self._ns_max - self._ns_min) / 2
        init_index = 2.19

        m = Minuit(
            self._func_to_minimize,
            ns=init_ns,
            index=init_index,
        )
        m.limits["ns"] = (self._ns_min, self._ns_max)
        m.errors["ns"] = 1
        m.errordef = 0.5
        m.migrad()

        self._best_fit_ns = m.values["ns"]

        return m

    def _minimize_grid(self):
        """
        Minimize -log(likelihood_ratio) for the source hypothesis,
        returning the best fit ns and index.

        This simple grid method takes roughly the same time as minuit
        and is more accurate...
        """

        ns_grid = np.linspace(self._ns_min, self._ns_max, 10)

        out = np.zeros(len(ns_grid))
        for i, ns in enumerate(ns_grid):
            out[i] = self._get_neg_log_likelihood_ratio(ns)

        sel = np.where(out == np.min(out))

        if len(sel[0]) > 1:

            self._best_fit_ns = 0.0

        else:

            self._best_fit_ns = ns_grid[sel[0]]

    def get_test_statistic(self):
        """
        Calculate the test statistic for the best fit ns
        """

        self._minimize()

        neg_log_lik = self._get_neg_log_likelihood_ratio(self._best_fit_ns)

        self.likelihood_ratio = np.exp(neg_log_lik)

        self.test_statistic = -2 * neg_log_lik

        return self.test_statistic


class SimplePointSourceLikelihood:
    def __init__(self, direction_likelihood, event_coords, source_coord):
        """
        Point source likelihood with only spatial information.
        Testing out simple algorithms for evaluating TS.
        """

        self._direction_likelihood = direction_likelihood

        self._band_width = 3 * direction_likelihood._sigma

        self._event_coords = event_coords

        self._source_coord = source_coord

        self._select_declination_band()

        self.Ntot = len(self._event_coords)

    def _signal_likelihood(self, event_coord, source_coord):

        return self._direction_likelihood(event_coord, source_coord)

    def _background_likelihood(self):

        return 1 / (np.deg2rad(self._band_width * 2) * 2 * np.pi)

    def _select_declination_band(self):

        decs = np.array([_[1] for _ in self._event_coords])

        _, source_dec = self._source_coord

        dec_fac = np.deg2rad(self._band_width)

        selected = np.where(
            (decs >= source_dec - dec_fac) & (decs <= source_dec + dec_fac)
        )[0]

        self._selected = selected

        self._selected_event_coords = [
            (ec[0], ec[1])
            for ec in self._event_coords
            if (ec[1] >= source_dec - dec_fac) & (ec[1] <= source_dec + dec_fac)
        ]

        self.N = len(selected)

    def __call__(self, ns):

        log_likelihood = 0.0

        for i in range(self.N):

            signal = (ns / self.N) * self._signal_likelihood(
                self._selected_event_coords[i], self._source_coord
            )

            bg = (1 - (ns / self.N)) * self._background_likelihood()

            log_likelihood += np.log(signal + bg)

        return -log_likelihood


class SimpleWithEnergyPointSourceLikelihood:
    def __init__(
        self, direction_likelihood, energy_likelihood, event_coords, source_coord
    ):
        """
        Simple version of point source likelihood.
        Also including the energy dependence.
        Testing out simple algorithms for evaluating TS.
        """

        self._direction_likelihood = direction_likelihood

        self._energy_likelihood = energy_likelihood

        self._band_width = 3 * direction_likelihood._sigma

        self._event_coords = event_coords

        self._source_coord = source_coord

        self._select_declination_band()

        self.Ntot = len(self._event_coords)

        self._bg_index = 3.7

    def _signal_likelihood(self, event_coord, source_coord, energy, index):

        return self._direction_likelihood(
            event_coord, source_coord
        ) * self._energy_likelihood(energy, index)

    def _background_likelihood(self, energy):

        return (
            1
            / (np.deg2rad(self._band_width * 2) * 2 * np.pi)
            * self._energy_likelihood(energy, self._bg_index)
        )

    def _select_declination_band(self):

        decs = np.array([_[1] for _ in self._event_coords])

        _, source_dec = self._source_coord

        dec_fac = np.deg2rad(self._band_width)

        selected = np.where(
            (decs >= source_dec - dec_fac) & (decs <= source_dec + dec_fac)
        )[0]

        self._selected = selected

        self._selected_event_coords = [
            (ec[0], ec[1])
            for ec in self._event_coords
            if (ec[1] >= source_dec - dec_fac) & (ec[1] <= source_dec + dec_fac)
        ]

        self.N = len(selected)

    def __call__(self, ns):

        log_likelihood = 0.0

        for i in range(self.N):

            signal = (ns / self.N) * self._signal_likelihood(
                self._selected_event_coords[i], self._source_coord
            )

            bg = (1 - (ns / self.N)) * self._background_likelihood()

            log_likelihood += np.log(signal + bg)

        return -log_likelihood
