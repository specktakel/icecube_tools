import numpy as np

from typing import Sequence

from .energy_likelihood import MarginalisedEnergyLikelihood
from .spatial_likelihood import SpatialLikelihood
from ..utils.data import RealEvents
from ..detector.effective_area import EffectiveArea


class DataDrivenBackgroundLikelihood(MarginalisedEnergyLikelihood, SpatialLikelihood):

    def __init__(self, period, bins: Sequence[float]=None):
        self._period = period
        self._events = RealEvents.from_event_files(period, use_all=True)

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

    def __call__(self, energy, index, dec):
        """
        Calculate energy likelihood for given events
        index is dummy argument s.t. PointSourceLikelihood doesn't complain
        """

        log_ereco = np.log10(energy)
        sin_dec_idx = np.digitize(np.sin(dec), self._sin_dec_bins) - 1
        energy_idx = np.digitize(log_ereco, self._ereco_bins) - 1

        return self._likelihood[sin_dec_idx, energy_idx] / (2 * np.pi)
