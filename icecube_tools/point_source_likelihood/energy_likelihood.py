import numpy as np
from scipy.stats import rv_histogram
from scipy.interpolate import splev, splrep
from scipy.signal import convolve
from abc import ABC, abstractmethod
import h5py
from os.path import join
from typing import Sequence

from ..detector.detector import Detector, IceCube
from ..utils.data import RealEvents
from ..detector.effective_area import EffectiveArea
from ..detector.r2021 import R2021IRF


from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator

"""
Module to compute the IceCube energy likelihood
using publicly available information.

Based on the method described in:
Braun, J. et al., 2008. Methods for point source analysis 
in high energy neutrino telescopes. Astroparticle Physics, 
29(4), pp.299-305.

Currently well-defined for searches with
Northern sky muon neutrinos.
"""


class Spline:
    """Class to represent a 1D function spline using the PchipInterpolator
    class from scipy.

    The evaluate the spline, use the ``__call__`` method.

    Shamelessly copied from skyllh.
    """

    def __init__(self, f, x_binedges, norm=False, **kwargs):
        """Creates a new 1D function spline using the PchipInterpolator
        class from scipy.

        Parameters
        ----------
        f : (n_x,)-shaped 1D numpy ndarray
            The numpy ndarray holding the function values at the bin centers.
        x_binedges : (n_x+1,)-shaped 1D numpy ndarray
            The numpy ndarray holding the bin edges of the x-axis.
        norm : bool
            Whether to precalculate and save normalization internally.
        """
        super().__init__(**kwargs)

        self.x_binedges = np.copy(x_binedges)

        self.x_min = self.x_binedges[0]
        self.x_max = self.x_binedges[-1]

        x = self.x_binedges[:-1] + np.diff(self.x_binedges) / 2

        self.spl_f = PchipInterpolator(x, f, extrapolate=False)

        self.norm = None
        if norm:
            self.norm = quad(
                self.__call__, self.x_min, self.x_max, limit=200, full_output=1
            )[0]

    def __call__(self, x, oor_value=0):
        """Evaluates the spline at the given x values. For x-values
        outside the spline's range, the oor_value is returned.

        Parameters
        ----------
        x : (n_x,)-shaped 1D numpy ndarray
            The numpy ndarray holding the x values at which the spline should
            get evaluated.
        oor_value : float
            The value for out-of-range (oor) coordinates.

        Returns
        -------
        f : (n_x,)-shaped 1D numpy ndarray
            The numpy ndarray holding the evaluated values of the spline.
        """
        f = self.spl_f(x)
        f = np.where(np.isnan(f), oor_value, f)

        return f

    def evaluate(self, *args, **kwargs):
        """Alias for the __call__ method."""
        return self(*args, **kwargs)


class MarginalisedEnergyLikelihood(ABC):
    """
    Abstract base class for the marginalised energy likelihood.

    L = \int d Etrue P(Ereco | Etrue) P(Etrue | index) = P(Ereco | index).
    """

    @abstractmethod
    def __call__(self):
        """
        Return the value of the likelihood for a given Ereco and index.
        """

        pass


class MarginalisedSkyLLHLikeEnergyLikelihood(MarginalisedEnergyLikelihood):

    def __call__(self, index):

        index_idx = np.digitize(index, self._index_bins) - 1
        x1 = self._index_grid[index_idx]
        try:
            M1, a, b = self._cache[x1]
        except:
            x0 = self._index_grid[index_idx - 1]

            x2 = self._index_grid[index_idx + 1]

            spline0 = self.calc_likelihood_for_index(x0)
            spline1 = self.calc_likelihood_for_index(x1)
            spline2 = self.calc_likelihood_for_index(x2)
            M0 = spline0(self._logEreco)
            M1 = spline1(self._logEreco)
            M2 = spline2(self._logEreco)

            a = 0.5 * (M0 - 2.0 * M1 + M2) / self._delta_index_squared
            b = 0.5 * (M2 - M0) / self._delta_index

            self._cache[x1] = (M1, a, b)

        values = a * (index - x1) ** 2 + b * (index - x1) + M1

        return values

    def __init__(self, source, period, energies):

        self._cache = {}

        self._index_grid = np.arange(1.0, 5.05, 0.1)
        self._index_bins = np.arange(0.95, 5.06, 0.1)

        self._delta_index = np.diff(self._index_grid)[0]
        self._delta_index_squared = np.power(self._delta_index, 2.0)
        self._logEreco = np.log10(energies)
        self._min_index = 1.1
        self._max_index = 4.9
        self.period = period
        self.source = source
        self.dec = source._coord[1]
        self.flux_model = source.flux_model
        self.common_ereco_bins = DataDrivenBackgroundEnergyLikelihood.LOG_ENERGY_BINS[
            self.period
        ]
        self.common_ereco_bin_cen = (
            self.common_ereco_bins[:-1] + np.diff(self.common_ereco_bins) / 2
        )

        self.setup()

    def setup(self):
        self.aeff = EffectiveArea.from_dataset("20210126", self.period)
        self.irf = R2021IRF.from_period(self.period)
        faulty_bins = self.irf.faulty
        self.dec_idx = np.digitize(self.dec, self.irf.declination_bins) - 1
        faulty_E = []
        for faulty in faulty_bins:
            if faulty[1] == self.dec_idx:
                faulty_E.append(faulty[0])
        try:
            logEtrue_bins = self.irf.true_energy_bins[max(faulty_E) + 1 :]
        except ValueError:
            logEtrue_bins = self.irf.true_energy_bins
        # Assume that only at low energies we find empty histograms
        self.Etrue_bins = np.power(10, logEtrue_bins)
        self.log_Etrue = logEtrue_bins[:-1] + np.diff(logEtrue_bins) / 2
        self.Etrue_idx = np.digitize(self.log_Etrue, self.irf.true_energy_bins) - 1

        print(logEtrue_bins)
        self.detection_prob = self.aeff.get_splined_detection_probability(
            self.Etrue_bins[:-1], self.Etrue_bins[1:], self.dec
        )

    def calc_likelihood_for_index(self, index):
        self.flux_model._index = index
        flux_norm = self.flux_model.integrated_spectrum(
            self.Etrue_bins[0], self.Etrue_bins[-1]
        )
        prob_per_bin = (
            self.flux_model.integrated_spectrum(
                self.Etrue_bins[:-1], self.Etrue_bins[1:]
            )
            / flux_norm
        )

        assert np.isclose(prob_per_bin.sum(), 1.0)

        p = prob_per_bin * self.detection_prob
        etrue_prob_per_bin = p / np.sum(p)

        assert np.isclose(etrue_prob_per_bin.sum(), 1.0)

        splines = []

        for i, tE_idx in enumerate(self.Etrue_idx):
            ereco_pdf = self.irf.reco_energy[tE_idx, self.dec_idx]
            bin_edges = self.irf.reco_energy_bins[tE_idx, self.dec_idx]
            bin_centers = self.irf.reco_energy_bin_cen[tE_idx, self.dec_idx]
            prob = ereco_pdf.pdf(bin_centers) * etrue_prob_per_bin[i]

            spline = Spline(prob, bin_edges)
            splines.append(spline(self.common_ereco_bin_cen))

        summed_spline = np.sum(splines, axis=0)

        spline_of_sum = Spline(summed_spline, self.common_ereco_bins, norm=True)

        return spline_of_sum


class MarginalisedIntegratedEnergyLikelihood(MarginalisedEnergyLikelihood):
    """
    Calculates energy likelihood by integration rather than simulation.
    """

    # @profile
    def __init__(
        self,
        # detector: Detector,
        period: str,
        reco_bins: np.ndarray,
        min_index: float = 1.5,
        max_index: float = 4.0,
    ):
        """
        Init likelihood.
        :param irf: Instance of :class:`icecube_tools.detector.r2021.R2021IRF`
        :param aeff: Instance of :class:`icecube_tools.detector.effective_area.EffectiveArea`
        :param reco_bins: Array of new reconstructed energy bins at which the likelihood is evaluated
        :param min_index: Smallest spectral index considered
        :param max_index: Largest spectral index considered
        """

        # TODO change reco_bins to cover the range provided by all the pdfs
        # and have the coarsest binning of all pdfs
        detector = IceCube.from_period(period)
        aeff = detector._effective_area
        irf = detector._angular_resolution
        self._irf = irf
        self._aeff = aeff
        self.reco_bins = reco_bins
        self.reco_centers = reco_bins[:-1] + np.diff(reco_bins) / 2
        self._irf_period = period
        # print(self.reco_bins)
        self.true_bins_irf = irf.true_energy_bins
        self.true_bins_aeff = np.log10(aeff.true_energy_bins)
        self.true_energy_bins = np.array(
            sorted(list(set(self.true_bins_irf).union(self.true_bins_aeff)))
        )
        idx = np.nonzero(
            (self.true_energy_bins <= self.true_bins_irf.max())
            & (self.true_energy_bins <= self.true_bins_aeff.max())
            & (self.true_energy_bins >= self.true_bins_irf.min())
            & (self.true_energy_bins >= self.true_bins_aeff.min())
        )
        self.true_energy_bins = self.true_energy_bins[idx]
        self.declination_bins_irf = irf.declination_bins
        self.cos_z_bins = aeff.cos_zenith_bins
        self.declination_bins_aeff = np.flip(np.arcsin(-self.cos_z_bins))
        self._min_index = min_index
        self._max_index = max_index
        self.true_bins_c = self.true_energy_bins[:-1] + 0.5 * np.diff(
            self.true_energy_bins
        )
        self._previous_index = None
        self._values = {}
        if self._irf_period == "IC86_II":
            self._events = RealEvents.from_event_files("IC86_II", use_all=True)
        else:
            self._events = RealEvents.from_event_files(self._irf_period)
        self._get_ereco_cuts()

        # pre-calculate cdf values
        self._cdf = np.zeros(
            (self.true_energy_bins.size - 1, 3, self.reco_bins.size - 1)
        )
        for c_true, e_true in enumerate(self.true_energy_bins[:-1]):
            c_irf_true = np.digitize(e_true, self.true_bins_irf) - 1
            for c_dec, _ in enumerate(self.declination_bins_irf[:-1]):
                for c, (erecol, erecoh) in enumerate(
                    zip(self.reco_bins[:-1], self.reco_bins[1:])
                ):
                    pdf = self._irf.reco_energy[c_irf_true, c_dec]
                    self._cdf[c_true, c_dec, c] = pdf.cdf(erecoh) - pdf.cdf(erecol)

    # @profile
    def __call__(self, ereco: np.ndarray, index: float, dec: np.ndarray) -> np.ndarray:
        """
        Wrapper on _calc_likelihood to retrieve only the likelihood for a specific Ereco value.
        Saves time by storing data and checking if data of the same index is requested
        over and over again, as is done in point_source_likelihood.py for each event.
        :param ereco: Reconstructed energy in GeV, float or np.ndarray
        :param index: Spectral index > 0
        :param dec: Declination, rad
        :return: Likelihood of reconstructed energy index at declination.
        """

        if index > self._max_index:
            raise ValueError("Index too high")
        elif index < self._min_index:
            raise ValueError("Index too low")

        log_ereco = np.log10(ereco)
        # print(log_ereco)
        reco_ind = np.digitize(log_ereco, self.reco_bins) - 1  # is np.ndarray
        # print(reco_ind)
        ok_ind = np.nonzero(((reco_ind >= 0) & (reco_ind < self.reco_bins.size - 1)))
        # print(ok_ind)
        reco_ind = reco_ind[ok_ind]  # reduce to those inside the provided energies
        # print(reco_ind)
        dec = dec[ok_ind]  # apply mask to declination as well
        dec_ind = np.digitize(dec, self.declination_bins_aeff) - 1  # is np.ndarray
        dec_ind_set = set(dec_ind)

        # output array, one entry for each queried ereco
        output = np.zeros_like(
            log_ereco
        )  # not-ok energies have zero probability returned, log is someone else's problem
        # loop over set(sec_ind):
        for dec_idx in dec_ind_set:
            # get declination of index
            single_dec = self.declination_bins_aeff[dec_idx]
            if dec_idx == 0:
                single_dec += 0.01  # necessary bc of np.digitize's left/right,
                # would lead to evaluation of upper bound in flipped array -> forbidden
            # for the queried dec index, calculate the likelihood
            self._values[dec_idx] = self._calc_likelihood(index, single_dec)
            # TODO: Insert a spline representation here
            needed = np.nonzero((dec_ind == dec_idx))
            output[needed] = splev(log_ereco[needed], self._values[dec_idx])

        return output

    # @profile
    def _calc_likelihood(self, index: float, dec: float) -> np.ndarray:
        """
        Calculates likelihood for given index at given declination.
        :param index: Spectral index
        :param dec: Declination in rad
        :return: Likelihood for each reco_bin
        """

        irf_dec_ind = np.digitize(dec, self.declination_bins_irf) - 1

        # pre-calculate power law and aeff part, is not dependent on reco energy
        # pl = np.zeros(self.true_energy_bins.size - 1)
        # for c, (etruel, etrueh) in enumerate(zip(
        #        self.true_energy_bins[:-1], self.true_energy_bins[1:])
        #    ):
        ##
        #    pl[c] = self.integrated_power_law(etrueh, etruel, index)
        pl = self.integrated_power_law(
            self.true_energy_bins[:-1], self.true_energy_bins[1:], index
        )

        aeff = self._aeff.detection_probability(
            np.power(10, self.true_bins_c), -np.sin(dec), 1e9
        )
        # one output value for each reco_bin (provided by some array at instantiation)
        values = np.zeros(self.reco_bins.size - 1)
        for c_reco, (erecol, erecoh) in enumerate(
            zip(self.reco_bins[:-1], self.reco_bins[1:])
        ):

            # Can this be done in without the loop?
            # integrate over true energy
            # print("pl", pl)
            sum_this = pl * self._cdf[:, irf_dec_ind, c_reco]
            # print("cdf", self._cdf[:, irf_dec_ind, c_reco])
            # print("pl*cdf", sum_this)
            values[c_reco] = np.dot(sum_this, aeff)
            # print("aeff*(pl*cdf)", values[c_reco])
        # print("values", values)
        norm = np.sum(values * np.diff(self.reco_bins))
        # print("norm", norm)
        values = values / norm
        tck = splrep(self.reco_centers, values)
        # return values
        return tck

    def _get_ereco_cuts(self):
        """
        Get ereco cuts from data, stores log10(Ereco).
        """

        self._ereco_limits = np.zeros((self.declination_bins_aeff.size - 1, 2))
        for c, (dec_low, dec_high) in enumerate(
            zip(self.declination_bins_aeff[:-1], self.declination_bins_aeff[1:])
        ):
            self._events.restrict(dec_low=dec_low, dec_high=dec_high)
            ereco = self._events.reco_energy[self._irf_period]
            self._ereco_limits[c, 0] = np.log10(ereco.min())
            self._ereco_limits[c, 1] = np.log10(ereco.max())

    def p_det_above_threshold(self, Etrue, dec):
        """
        Calculate probability of an event with Etrue being reconstructed
        above and below given thresholds, which are provided by the data
        and are declination dependent.
        """

        log_etrue = np.log10(Etrue)
        dec_idx_aeff = np.digitize(dec, self.declination_bins_aeff) - 1
        ereco_low = self._ereco_limits[dec_idx_aeff, 0]
        # ereco_high = self._ereco_limits[dec_idx_aeff, 1]
        # only use lower energy bound, as per the IceCubes

        dec_idx_irf = np.digitize(dec, self._irf.declination_bins) - 1
        etrue_idx = np.digitize(log_etrue, self._irf.true_energy_bins) - 1

        # pdf is self._irf.reco_energy[]
        cdf = self._irf.reco_energy[etrue_idx, dec_idx_irf].cdf
        return 1.0 - cdf(ereco_low)

    @staticmethod
    def integrated_power_law(loge_low, loge_high, index):
        """
        Integrates power law
        :param loge_low: float or np.ndarray of upper integration bound(s)
        :param loge_high: float or np.ndarray of lower integration bound(s)
        :param index: spectral index
        :return: Integrated power law, float or np.ndarray
        """
        # works with np.ndarrays!
        return (
            1.0
            / (1 - index)
            * (
                np.power(10, -loge_high * (index - 1))
                - np.power(10, -loge_low * (index - 1))
            )
        )

    @staticmethod
    def power_law_loge(loge, index):
        """
        Evaluated power law
        :param loge: Logarithmic energy, base 10
        :param index: Spectral index
        :return: Evaluated power law
        """

        return np.power(np.power(10, loge), -index + 1)


class SplinedEnergyLikelihood(MarginalisedEnergyLikelihood):

    def __init__(self, aeff, min_index=1.5, max_index=4.0):
        pass

    def __call__(self, energy):
        pass

    pass


class MarginalisedEnergyLikelihood2021(MarginalisedEnergyLikelihood):
    """
    Compute the marginalised energy likelihood by reading in the provided IRF data of 2021.
    Creates instances of MarginalisedEnergyLikelihoodFromSimFixedIndex (slightly copied from
    MarginalisedEnergyLikelihoodFromSim but with different interpolating) for each given index.
    """

    def __init__(
        self,
        index_list,
        path,
        fname,
        src_dec,
        ftype="h5",
        min_index=1.5,
        max_index=4.0,
        min_E=1e2,
        max_E=1e9,
        min_sind=-0.1,
        max_sind=1.0,
        Ebins=50,
    ):
        """
        Initialise all datasets
        User needs to make sure that data sets cover the entire declination needed.

        :param index_list: List of indices provided with datasets
        :param path: Path where datasets are located
        :param fname: Filename, bar ending of `_{index:.1f}.txt`
        :param src_dec: Source declination in radians
        """
        # TODO change path thing and loading of data? maybe option to pass data directly
        # for each index, load a different MarginalisedEnergyLikelihoodFromSim
        # distinguish between used data set/likelihood for different indices
        self.index_list = sorted(index_list)
        self.likelihood = {}

        for c, i in enumerate(self.index_list):
            filename = join(path, f"{fname}_index_{i:.1f}.h5")
            print(filename)
            with h5py.File(filename, "r") as f:
                reco_energy = f["reco_energy"][()]
                dec = f["dec"][()]
                # ang_err not needed
                # ang_err = f["ang_err"][()]
            self.likelihood[f"{float(i):1.1f}"] = (
                MarginalisedEnergyLikelihoodFromSimFixedIndex(
                    reco_energy,
                    dec,
                    i,
                    src_dec,
                    min_E,
                    max_E,
                    min_sind,
                    max_sind,
                    Ebins,
                )
            )
        self.lls = np.zeros(
            (
                len(index_list),
                self.likelihood[f"{float(i):1.1f}"]._energy_bins.shape[0] - 1,
            )
        )
        for c, i in enumerate(self.index_list):
            self.lls[c, :] = self.likelihood[f"{i:.1f}"].likelihood
            self._energy_bins = self.likelihood[f"{i:.1f}"]._energy_bins

        # decide on max/min index based on provided simulations
        # if range of simulations is smaller, use these values
        # else use user-provided values
        self._delta_index = 0.05

        if max_index > max(index_list) - self._delta_index:
            self._max_index = max(index_list) - self._delta_index
        else:
            self._max_index = max_index

        if min_index < min(index_list) + self._delta_index:
            self._min_index = min(index_list) + self._delta_index
        else:
            self._min_index = min_index

        self._min_E = min_E
        self._max_E = max_E
        self._min_sind = min_sind
        self._max_sind = max_sind
        self._Ebins = Ebins

    def __call__(self, E, index, dec=0):
        """
        Returns likelihood of reconstructed energy for specified spectral index.
        :param E: Reconstructed energy in GeV, may be float or np.ndarray
        :param index: spectral index
        :param dec: dummy argument
        :return: Likelihood
        :raise ValueError: if the requested index is out of range.
        :raise ValueError: if any other interpolation than `log` or `lin` is requested.
        """

        if index < min(self.index_list) or index > max(self.index_list):
            raise ValueError(f"Index {index} outside of range of index list.")

        if index not in self.index_list:
            raise ValueError("Only indices with simulation are allowed.")
        idx = np.digitize(np.log10(E), self._energy_bins) - 1

        index_index = np.digitize(index, self.index_list) - 1
        if index == max(self.index_list):
            index_index -= 1
        return self.lls[index_index, idx]

    def calc_loglike(self, energies, index):
        """
        Function intended for testing only.
        """

        loglike = 0
        self.faulty = []
        for e in energies:
            temp = self.__call__(e, index)
            if temp == 0.0:
                self.faulty.append(e)
                temp = 1e-10
            loglike += np.log10(temp)

        return -loglike


class DataDrivenBackgroundEnergyLikelihood(MarginalisedEnergyLikelihood):
    """
    Energy likelihood for background obtained by making a distribution
    from the reconstructed energies. Data is mostly background.
    No spectral index is assumed.
    """

    b = np.sin(np.radians(-5.0))  # North/South transition boundary.
    # Binning copied from SkyLLH.datasets.i3.PublicData_10y_ps.create_dataset_collection
    SIN_DEC_BINS = {
        "IC40": np.unique(
            np.concatenate(
                [
                    np.linspace(-1.0, -0.25, 10 + 1),
                    np.linspace(-0.25, 0.0, 10 + 1),
                    np.linspace(0.0, 1.0, 10 + 1),
                ]
            )
        ),
        "IC59": np.unique(
            np.concatenate(
                [
                    np.linspace(-1.0, -0.95, 2 + 1),
                    np.linspace(-0.95, -0.25, 25 + 1),
                    np.linspace(-0.25, 0.05, 15 + 1),
                    np.linspace(0.05, 1.0, 10 + 1),
                ]
            )
        ),
        "IC79": np.unique(
            np.concatenate(
                [
                    np.linspace(-1.0, -0.75, 10 + 1),
                    np.linspace(-0.75, 0.0, 15 + 1),
                    np.linspace(0.0, 1.0, 20 + 1),
                ]
            )
        ),
        "IC86_I": np.unique(
            np.concatenate(
                [
                    np.linspace(-1.0, -0.2, 10 + 1),
                    np.linspace(-0.2, b, 4 + 1),
                    np.linspace(b, 0.2, 5 + 1),
                    np.linspace(0.2, 1.0, 10),
                ]
            )
        ),
        "IC86_II": np.unique(
            np.concatenate(
                [
                    np.linspace(-1.0, -0.93, 4 + 1),
                    np.linspace(-0.93, -0.3, 10 + 1),
                    np.linspace(-0.3, 0.05, 9 + 1),
                    np.linspace(0.05, 1.0, 18 + 1),
                ]
            )
        ),
    }

    LOG_ENERGY_BINS = {
        "IC40": np.arange(2.0, 9.5 + 0.01, 0.125),
        "IC59": np.arange(2.0, 9.5 + 0.01, 0.125),
        "IC79": np.arange(2.0, 9.5 + 0.01, 0.125),
        "IC86_I": np.arange(1.0, 10.5 + 0.01, 0.125),
        "IC86_II": np.arange(0.5, 9.5 + 0.01, 0.125),
    }

    SPLINE_DEGREE = 2

    def __init__(self, period):
        self._period = period
        self._events = RealEvents.from_event_files(period, use_all=True)

        # Combine declination bins of the irf and aeff
        # self._sin_dec_aeff_bins = np.linspace(-1., 1., num=51, endpoint=True)
        # aeff = EffectiveArea.from_dataset("20210126", period)
        # osz_bins = aeff.cos_zenith_bins
        # self._sin_dec_bins = np.sort(-cosz_bins)
        # self._dec_bins = np.arcsin(self._sin_dec_bins)
        # self._declination_bin_edges = np.sort(self._dec_aeff_bins) # np.sort(np.union1d(np.deg2rad([-90, -10, 10, 90]), self._dec_aeff_bins))

        # Again, copied from SkyLLH
        self._sin_dec_bins = self.SIN_DEC_BINS[self._period]
        self._ereco_bins = self.LOG_ENERGY_BINS[self._period]
        log10_Ereco = np.log10(self._events.reco_energy[self._period])
        sin_dec = np.sin(self._events.dec[self._period])
        h, _, _ = np.histogram2d(
            log10_Ereco,
            sin_dec,
            bins=[self._ereco_bins, self._sin_dec_bins],
            density=False,
        )

        norms = (
            np.sum(h, axis=(0,))[np.newaxis, ...]
            * np.diff(self.LOG_ENERGY_BINS[self._period])[..., np.newaxis]
        )
        h /= norms

        # Smooth over energy by averaging a bin with its neighbours using scipy.signal.convolve
        # Copied from skyllh
        # Hardcoded defaults from the i3 analysis script
        kernel = np.prod(np.meshgrid(np.ones(3), np.ones(1), indexing="ij"), axis=0)
        norm = convolve(np.ones_like(h), kernel, mode="same")
        hists = convolve(h, kernel, mode="same") / norm

        self._hists = hists

        """
        if bins is None:
            self._ereco_bins = np.linspace(1, 8, num=50)
        else:
            self._ereco_bins = bins
        self.make_hist()
        """

    def __call__(self, energy, dec):
        """
        Calculate energy likelihood for given events
        index is dummy argument s.t. PointSourceLikelihood doesn't complain
        """

        log_ereco = np.log10(energy)
        sin_dec_idx = np.digitize(np.sin(dec), self._sin_dec_bins) - 1
        energy_idx = np.digitize(log_ereco, self._ereco_bins) - 1

        return self._hists[energy_idx, sin_dec_idx]

    def make_hist(self):
        """
        Create pdf-histograms
        """

        self._likelihood = np.zeros(
            (self._sin_dec_bins.size - 1, self._ereco_bins.size - 1)
        )
        self._rv_histogram = []
        self._costheta_bin_edges = np.sort(np.cos(np.pi / 2 - self._dec_bins))
        # Use real data to create pdf of the cos(theta) distribution
        # Use cos(theta) for easier sampling on a sphere, is converted to dec in simulator
        self._costheta_rv_histogram = rv_histogram(
            np.histogram(
                np.cos(np.pi / 2 - self._events.dec[self._period]),
                self._costheta_bin_edges,
            ),
            density=True,
        )

        # Loop over declination bins and create ereco distribution for each bin
        # both sin(dec) and dec increase monotically, so one loop for both is fine
        for c, (dec_l, dec_h) in enumerate(
            zip(self._dec_bins[:-1], self._dec_bins[1:])
        ):
            self._events.restrict(dec_low=dec_l, dec_high=dec_h)
            llh, bins = np.histogram(
                np.log10(self._events.reco_energy[self._period]),
                bins=self._ereco_bins,
                density=True,
            )
            self._likelihood[c, :] = llh
            # If there are events in the declination bin, make an rv_histogram
            if np.any(llh):
                self._rv_histogram.append(rv_histogram((llh, bins), density=True))
            else:
                self._rv_histogram.append(0)
        self._events.restrict()
        # 1st value is declination, 2nd value is energy
        self.hist_2d, _, _ = np.histogram2d(
            self._events.dec[self._period],
            np.log10(self._events.reco_energy[self._period]),
            [self._sin_dec_bins, self._ereco_bins],
            density=True,
        )

    def sample(self, dec, seed=42):
        """
        Sample from pdfs
        :param dec: np.ndarray of declinations of events
        :return: Samples drawn from the pdfs of corresponding declination bin
        """

        output = np.zeros_like(dec)
        sin_dec_idx = np.digitize(np.sin(dec), self._sin_dec_bins) - 1
        for sd_c in range(self._sin_dec_bins.size - 1):
            idx = np.nonzero(sin_dec_idx == sd_c)
            size = idx[0].size
            output[idx] = self._rv_histogram[sd_c].rvs(size=size, random_state=seed)
        return output


class MarginalisedEnergyLikelihoodFromSimFixedIndex(MarginalisedEnergyLikelihood):
    """
    Copied from MarginalisedEnergyLikelihoodFromSim but without the interpolating
    """

    def __init__(
        self,
        energy,
        dec,
        sim_index,
        src_dec=0.0,
        min_E=1e2,
        max_E=1e9,
        min_sind=-0.1,
        max_sind=1.0,
        Ebins=50,
    ):
        """
        :param energy: List of reconstructed energies from simulatedevents
        :param dec: List of declinations from simulated events
        :param sim_index: Spectral index used for the simulated events
        :param src_dec: Declination of source to be analised, in radians
        """

        self._energy = energy
        self._dec = dec
        self._sim_index = sim_index
        self._min_E = min_E
        self._max_E = max_E
        self._min_sind = min_sind
        self._max_sind = max_sind
        self._energy_bins = np.linspace(np.log10(min_E), np.log10(max_E), Ebins)  # GeV
        self._sin_dec_bins = np.linspace(min_sind, max_sind, 20)
        self.src_dec = src_dec

    def __call__(self, E):
        """
        Return likelihood of some reconstructed energy.
        :param E: Reconstructed energy in GeV, may be float or np.ndarray
        :return: Likelihood
        """

        idx = np.digitize(np.log10(E), self._energy_bins) - 1
        return self.likelihood[idx]

    @property
    def src_dec(self):
        return self._src_dec

    @src_dec.setter
    def src_dec(self, val):
        self._src_dec = val
        self._precompute_histograms()

    def _precompute_histograms(self):
        """
        Computes histograms of reconstructed energy from data set.
        Only uses data close in declination to specified source to account for declination dependence.
        """

        sind_idx = np.digitize(np.sin(self._src_dec), self._sin_dec_bins) - 1
        idx = (np.sin(self._dec) >= self._sin_dec_bins[sind_idx]) & (
            np.sin(self._dec) < self._sin_dec_bins[sind_idx + 1]
        )
        self._selected_energy = self._energy[idx]
        self.likelihood, _ = np.histogram(
            np.log10(self._selected_energy), bins=self._energy_bins, density=True
        )


class MarginalisedEnergyLikelihoodFromSim(MarginalisedEnergyLikelihood):
    """
    Compute the marginalised energy likelihood by using a
    simulation of a large number of reconstructed muon
    neutrino tracks.
    """

    def __init__(
        self,
        energy,
        dec,
        sim_index=1.5,
        min_index=1.5,
        max_index=4.0,
        min_E=1e2,
        max_E=1e9,
        min_sind=-0.1,
        max_sind=1.0,
        Ebins=50,
    ):
        """
        Compute the marginalised energy likelihood by using a
        simulation of a large number of reconstructed muon
        neutrino tracks.

        :param energy: Reconstructed muon energies (preferably many).
        :param dec: Reconstrcuted decs corresponding to these events.
        :param sim_index: Spectral index of source spectrum in sim.
        """

        self._energy = energy

        self._dec = dec

        self._sim_index = sim_index

        self._min_index = min_index
        self._max_index = max_index

        self._min_E = min_E
        self._max_E = max_E

        self._index_bins = np.linspace(min_index, max_index)

        self._energy_bins = np.linspace(np.log10(min_E), np.log10(max_E), Ebins)  # GeV

        self._sin_dec_bins = np.linspace(min_sind, max_sind, 20)

        self._src_dec = None

    def set_src_dec(self, src_dec):
        """
        Set the source declination in private variable
        Precompute likelihood distributions
        """

        self._src_dec = src_dec

        self._precompute_histograms()

    def _calc_weights(self, new_index):
        """
        Only compute one simulation with some given spectral index, num=index_bins.
        In order to test against multiple indices, re-use existing data
        but give new weights according to new spectral index.
        If |gamma| > |simulated gamma|, less weight needs to be at higher energies
        Convetion: index is positive, minus is explicitely stated in equations,
        assume flat spectrum (gamma=0) simulated:
        for gamma=2, shift to lower energies -> self._sim_index - new_index = 0 - 2 = -2
        -> the higher the energy, the lower the weight!
        """

        return np.power(self._selected_energy, self._sim_index - new_index)

    def _precompute_histograms(self):
        """
        Creates histograms of for each tested spectral index:
        self._likelihood empty list, one entry for each spectral index
        index of sin(dec) of source in list of sin(dec) sourrounding source
        energy is list of all Ereco from simulated events, index (idx) those who belong to correct declinations
        _selected_energy then contains all Ereco belonging to the selected events
        get index bin center
        create histogram (i.e. probability of finding some Ereco for given spectral index) for each spectral index
        """
        # TODO maybe change the sin(dec) bins to something more like +/- specified range?
        # what if src dec is right at a bin edge? too many events discarded!
        self._likelihood = np.zeros(
            (len(self._index_bins[:-1]), len(self._energy_bins[:-1]))
        )

        sind_idx = np.digitize(np.sin(self._src_dec), self._sin_dec_bins) - 1

        # only use events within the declination band hosting the source
        idx = (np.sin(self._dec) >= self._sin_dec_bins[sind_idx]) & (
            np.sin(self._dec) < self._sin_dec_bins[sind_idx + 1]
        )

        self._selected_energy = self._energy[idx]

        for i, index in enumerate(self._index_bins[:-1]):

            index_cen = index + (self._index_bins[i + 1] - index) / 2

            weights = self._calc_weights(index_cen)

            hist, _ = np.histogram(
                np.log10(self._selected_energy),
                bins=self._energy_bins,
                weights=weights,
                density=True,
            )

            self._likelihood[i] = hist

    def __call__(self, E, new_index, dec):
        """
        P(Ereco | index) = \int dEtrue P(Ereco | Etrue) P(Etrue | index)
        """

        # check for E out of bounds
        if E < self._min_E or E > self._max_E:

            raise ValueError(
                "Energy "
                + str(E)
                + "is not in the accepted range between "
                + str(self._min_E)
                + " and "
                + str(self._max_E)
            )

        # check for index out of bounds
        if new_index < self._min_index or new_index > self._max_index:

            raise ValueError(
                "Sepctral index "
                + str(new_index)
                + " is not in the accepted range between "
                + str(self._min_index)
                + " and "
                + str(self._max_index)
            )

        i_index = np.digitize(new_index, self._index_bins) - 1

        E_index = np.digitize(np.log10(E), self._energy_bins) - 1

        return self._likelihood[i_index][E_index]


class MarginalisedEnergyLikelihoodFixed(MarginalisedEnergyLikelihood):
    """
    Compute the marginalised energy likelihood for a fixed case based on a simulation.
    Eg. P(E | atmos + diffuse astro).
    """

    def __init__(
        self,
        energy,
        min_index=1.5,
        max_index=4.0,
        min_E=1e2,
        max_E=1e9,
        min_sind=-0.1,
        max_sind=1.0,
        Ebins=50,
    ):
        """
        Compute the marginalised energy likelihood for a fixed case based on a simulation.
        Eg. P(E | atmos + diffuse astro).

        :param energy: Reconstructed muon energies (preferably many) [GeV].
        """

        self._energy = energy

        self._min_index = min_index
        self._max_index = max_index

        self._min_E = min_E
        self._max_E = max_E

        self._energy_bins = np.linspace(np.log10(min_E), np.log10(max_E), Ebins)  # GeV

        self._precompute_histogram()

    def _precompute_histogram(self):

        hist, _ = np.histogram(
            np.log10(self._energy), bins=self._energy_bins, density=True
        )

        self._likelihood = hist

    def __call__(self, E):

        E_index = np.digitize(np.log10(E), self._energy_bins) - 1

        return self._likelihood[E_index]


class MarginalisedEnergyLikelihoodBraun2008(MarginalisedEnergyLikelihood):
    """
    Compute the marginalised enegry likelihood using
    Figure 4 in Braun+2008.
    """

    def __init__(self, energy_list, pdf_list, index_list):
        """
        Compute the marginalised enegry likelihood using
        Figure 4 in Braun+2008.

        :param energy_list: list of Ereco values (x-axis)
        :param pdf_list: list of P(Ereco | index) values (y-axis)
        :param index_list: list of spectral index inputs (diff lines)
        """

        self._energy_list = energy_list

        self._pdf_list = pdf_list

        self._index_list = index_list

        self._min_index = min(index_list)

        self._max_index = max(index_list)

    def __call__(self, energy, index, dec=0):
        """
        Return P(Ereco | index)
        """

        pdf_vals_at_E = [
            np.interp(energy, e, p) for e, p in zip(self._energy_list, self._pdf_list)
        ]

        pdf_val_at_index = np.interp(index, self._index_list, pdf_vals_at_E)

        return pdf_val_at_index


def reweight_spectrum(energies, sim_index, new_index, bins=int(1e3)):
    """
    Use energies from a simulation with a harder
    spectral index for efficiency.

    The spectrum is then resampled from the
    weighted histogram

    :param energies: Energies to be reiweghted.
    :sim_index: Spectral index of the simulation.
    :new_index: Spectral index to reweight to.
    """

    weights = np.array([np.power(_, sim_index - new_index) for _ in energies])

    hist, bins = np.histogram(
        np.log10(energies), bins=bins, weights=weights, density=True
    )

    bin_midpoints = bins[:-1] + np.diff(bins) / 2

    cdf = np.cumsum(hist)
    cdf = cdf / float(cdf[-1])

    values = np.random.rand(len(energies))

    value_bins = np.searchsorted(cdf, values)

    random_from_cdf = bin_midpoints[value_bins]

    energies = np.power(10, random_from_cdf)

    return energies


def read_input_from_file(filename):
    """
    Helper function to read in data digitized from plots.
    """

    import h5py

    keys = ["E-2_spectrum", "E-2.5_spectrum", "E-3_spectrum", "atmospheric"]

    index_list = []
    energy_list = []
    pdf_list = []

    with h5py.File(filename, "r") as f:

        for key in keys:

            index_list.append(f[key]["index"][()])

            energy_list.append(f[key]["reco_energy"][()])

            pdf_list.append(f[key]["pdf"][()])

    return energy_list, pdf_list, index_list
