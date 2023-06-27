import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import gaussian_kde
from typing import Sequence

from .energy_likelihood import MarginalisedEnergyLikelihood
from .spatial_likelihood import SpatialLikelihood
from ..utils.data import RealEvents
from ..detector.effective_area import EffectiveArea


class DataDrivenBackgroundLikelihood(MarginalisedEnergyLikelihood, SpatialLikelihood):

    def __init__(self, period, bins: Sequence[float] = None, spline: bool = True, kde: bool = False):
        self._period = period
        self._events = RealEvents.from_event_files(period, use_all=True)
        self._spline = spline
        self._kde = kde

        # Combine declination bins of the irf and aeff
        # self._sin_dec_aeff_bins = np.linspace(-1., 1., num=51, endpoint=True)
        aeff = EffectiveArea.from_dataset("20210126", period)

        if bins is None:
            self._ereco_bins = np.linspace(1, 9, num=50)
        else:
            self._ereco_bins = bins
        cosz_bins = aeff.cos_zenith_bins

        self._sin_dec_bins = np.sort(-cosz_bins)
        self._dec_bins = np.arcsin(self._sin_dec_bins)

        self._likelihood, _, _ = np.histogram2d(
            np.sin(self._events.dec[self._period]),
            np.log10(self._events.reco_energy[self._period]),
            [self._sin_dec_bins, self._ereco_bins],
            density=True
        )

        if spline:
            self._spline_this = self._likelihood.copy()
            # Do this to avoid nans and infs
            self._spline_this[self._likelihood == 0.] = 1e-10
            self._sin_dec_bins_c = (self._sin_dec_bins[:-1] + self._sin_dec_bins[1:]) / 2
            self._ereco_bins_c = (self._ereco_bins[:-1] + self._ereco_bins[1:]) / 2
            self._splined_llh = RegularGridInterpolator(
                (
                    self._sin_dec_bins_c,
                    self._ereco_bins_c,
                ),
                np.log10(self._spline_this),
                bounds_error=False
            )

        elif kde:
            ereco = np.log10(self._events.reco_energy[self._period])
            sin_dec = np.sin(self._events.dec[self._period])
            self._kde_likelihood = gaussian_kde(np.vstack((ereco, sin_dec)))

            

    def __call__(self, energy, index, dec):
        """
        Calculate energy likelihood for given events
        index is dummy argument s.t. PointSourceLikelihood doesn't complain
        """

        log_ereco = np.log10(energy)
        

        if self._spline:
            dec = np.atleast_1d(dec)
            log_ereco = np.atleast_1d(log_ereco)
            coords = np.vstack((np.sin(dec), log_ereco)).T
            return np.power(10, self._splined_llh(coords)) / (2 * np.pi)

        elif self._kde:

            return self._kde_likelihood(np.vstack((log_ereco, np.sin(dec)))) / (2 * np.pi)

        else:
            sin_dec_idx = np.digitize(np.sin(dec), self._sin_dec_bins) - 1
            energy_idx = np.digitize(log_ereco, self._ereco_bins) - 1
            return self._likelihood[sin_dec_idx, energy_idx] / (2 * np.pi)
