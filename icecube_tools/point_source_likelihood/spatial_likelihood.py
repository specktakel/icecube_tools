import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple
from scipy.stats import rv_histogram
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from ..utils.data import RealEvents
from ..detector.effective_area import EffectiveArea


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


class SpatialLikelihood(ABC):
    """
    Abstract base class for spatial likelihoods
    """

    @abstractmethod
    def __call__(self):

        pass


class RayleighDistribution(SpatialLikelihood):
    """
    If you transform a 2d Gauss from cartesian to polar coordinates,
    you'd probably arrive here.
    """

    def __init__(self, sigma=2):
        # Why am I copying this? Is this used somewhere?
        self._sigma = sigma

    def __call__(
        self,
        ang_err: np.ndarray,
        ra: np.ndarray,
        dec: np.ndarray,
        source_coord: Tuple[float, float],
    ):
        """
        Returns the Rayleigh distribution after integrating over phi.
        """

        sigma_rad = np.deg2rad(ang_err)

        src_ra, src_dec = source_coord

        # Calculate the cosine of the distance of the source and the event on
        # the sphere.
        cos_r = np.cos(src_ra - ra) * np.cos(src_dec) * np.cos(dec) + np.sin(
            src_dec
        ) * np.sin(dec)

        # Handle possible floating precision errors.
        idx = np.nonzero((cos_r < -1.0))
        cos_r[idx] = 1.0
        idx = np.nonzero((cos_r > 1.0))
        cos_r[idx] = 1.0

        r = np.arccos(cos_r)

        return (
            r
            / (2 * np.pi * np.sin(r) * np.power(sigma_rad, 2))
            * np.exp(-np.power(r, 2) / (2 * np.power(sigma_rad, 2)))
        )


class EventDependentSpatialGaussianLikelihood(SpatialLikelihood):
    def __init__(self, sigma=2):
        """
        :param sigma: Upper limit of angular distance to considered events
        """
        # TODO actually implement this somehow
        # convert p to sigma if necessary...
        # omit for now
        self._sigma = sigma

    # @profile
    def __call__(
        self,
        ang_err: np.ndarray,
        ra: np.ndarray,
        dec: np.ndarray,
        source_coord: Tuple[float, float],
    ):
        """
        Use the neutrino energy to determine sigma and
        evaluate the likelihood.

        P(x_i | x_s) = (1 / (2pi * sigma^2)) * exp( |x_i - x_s|^2/ (2*sigma^2) )

        :param ang_err: Angular error to be used in the Gaussian, in degrees
        :param ra: RAs of events, in rad
        :param dec: DECs of events, in rad
        :param source_coord: Tuple (ra, dec) of point source [rad].
        """

        sigma_rad = np.deg2rad(ang_err)

        src_ra, src_dec = source_coord

        norm = 0.5 / (np.pi * sigma_rad**2)

        # Calculate the cosine of the distance of the source and the event on
        # the sphere.
        cos_r = np.cos(src_ra - ra) * np.cos(src_dec) * np.cos(dec) + np.sin(
            src_dec
        ) * np.sin(dec)

        # Handle possible floating precision errors.
        idx = np.nonzero((cos_r < -1.0))
        cos_r[idx] = 1.0
        idx = np.nonzero((cos_r > 1.0))
        cos_r[idx] = 1.0

        r = np.arccos(cos_r)

        dist = np.exp(-0.5 * (r / sigma_rad) ** 2)

        return r / np.sin(r) * norm * dist


class DataDrivenBackgroundSpatialLikelihood(SpatialLikelihood):
    """
    Class using data to calculate the background spatial likelihood,
    instead of using 1 / (4pi) as one would naively use
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

    def __init__(self, period: str):

        self._period = period
        self._events = RealEvents.from_event_files(period, use_all=True)
        aeff = EffectiveArea.from_dataset("20210126", period)
        cosz_bins = aeff.cos_zenith_bins
        # self._sin_dec_bins = np.sort(-cosz_bins)
        self._sin_dec_bins = self.SIN_DEC_BINS[self._period]
        self._dec_bins = np.arcsin(self._sin_dec_bins)
        self._hist, _ = np.histogram(
            np.sin(self._events.dec[self._period]),
            bins=self._sin_dec_bins,
            density=True,
        )
        # self.likelihood = rv_histogram(self.hist, density=True)
        self._sin_dec_bin_centers = (
            self._sin_dec_bins[:-1] + np.diff(self._sin_dec_bins) / 2
        )
        self._spline = Spline(
            self._sin_dec_bin_centers, np.log(self._hist), k=self.SPLINE_DEGREE
        )

    def __call__(self, dec: np.ndarray):
        """
        Returns likelihood for each provided event, 2pi accounts for uniform RA distribution.
        """

        return np.exp(self._spline(np.sin(dec))) / (2 * np.pi)


class SpatialGaussianLikelihood(SpatialLikelihood):
    """
    Spatial part of the point source likelihood.

    P(x_i | x_s) where x is the direction (unit_vector).
    """

    def __init__(self, angular_resolution):
        """
        Spatial part of the point source likelihood.

        P(x_i | x_s) where x is the direction (unit_vector).

        :param angular_resolution; Angular resolution of detector [deg].
        """

        # @TODO: Init with some sigma as a function of E?

        self._sigma = angular_resolution

    def __call__(self, ra, dec, source_coord):
        """
        Use the neutrino energy to determine sigma and
        evaluate the likelihood.

        P(x_i | x_s) = (1 / (2pi * sigma^2)) * exp( |x_i - x_s|^2/ (2*sigma^2) )

        :param event_coord: (ra, dec) of event [rad].
        :param source_coord: (ra, dec) of point source [rad].
        """

        sigma_rad = np.deg2rad(self._sigma)

        src_ra, src_dec = source_coord

        norm = 0.5 / (np.pi * sigma_rad**2)

        # Calculate the cosine of the distance of the source and the event on
        # the sphere.
        cos_r = np.cos(src_ra - ra) * np.cos(src_dec) * np.cos(dec) + np.sin(
            src_dec
        ) * np.sin(dec)

        # Handle possible floating precision errors.
        idx = np.nonzero((cos_r < -1.0))
        cos_r[idx] = 1.0
        idx = np.nonzero((cos_r > 1.0))
        cos_r[idx] = 1.0

        r = np.arccos(cos_r)

        dist = np.exp(-0.5 * (r / sigma_rad) ** 2)

        return norm * dist


class EnergyDependentSpatialGaussianLikelihood(SpatialLikelihood):
    """
    Energy dependent spatial likelihood. Uses AngularResolution
    specified for given spectral indicies. For example in the 2015
    data release angular resolution plots.

    The atmospheric spectrum is approximated as a power law
    with a single spectral index.
    """

    def __init__(self, angular_resolution_list, index_list):
        """
        Energy dependent spatial likelihood. Uses AngularResolution
        specified for given spectral indicies. For example in the 2015
        data release angular resolution plots.

        The atmospheric spectrum is approximated as a power law
        with a single spectral index.

        :param angular_resolution_list: List of AngularResolution instances.
        :param index_list: List of corresponding spectral indices
        """

        self._angular_resolution_list = angular_resolution_list

        self._index_list = index_list

    def _get_sigma(self, reco_energy, index):
        """
        Return the expected angular resolution for a
        given reconstrcuted energy and spectral index.

        :param reco_energy: Reconstructed energy [GeV]
        :param index: Spectral index
        """

        ang_res_at_Ereco = [
            ang_res._get_angular_resolution(reco_energy)
            for ang_res in self._angular_resolution_list
        ]

        ang_res_at_index = np.interp(index, self._index_list, ang_res_at_Ereco)

        return ang_res_at_index

    def get_low_res(self):
        """
        Representative lower resolution
        at fixed low energy and bg index.

        To be used in PointSourceLikelihood.
        """

        low_energy = 1e3

        bg_index = 3.7

        return self._get_sigma(low_energy, bg_index)

    def __call__(self, ra, dec, source_coord, reco_energy, index=2.0):
        """
        Evaluate PDF for a given event.

        :param event_coord: (ra, dec) coordinates of event
        :param source_coord: (ra, dec) coordinates of source
        :param reco_energy: Reconstructed energy [GeV]
        :param index: Spectral index of source
        """
        output = np.zeros_like(ra)
        src_ra, src_dec = source_coord
        for c, (r, d, e) in enumerate(zip(ra, dec, reco_energy)):

            sigma_rad = np.deg2rad(self._get_sigma(e, index))

            norm = 0.5 / (np.pi * sigma_rad**2)

            # Calculate the cosine of the distance of the source and the event on
            # the sphere.
            cos_r = np.cos(src_ra - r) * np.cos(src_dec) * np.cos(d) + np.sin(
                src_dec
            ) * np.sin(d)

            # Handle possible floating precision errors.
            if cos_r < -1.0:
                cos_r = 1.0
            if cos_r > 1.0:
                cos_r = 1.0

            r = np.arccos(cos_r)

            dist = np.exp(-0.5 * (r / sigma_rad) ** 2)

            output[c] = dist * norm

        return output
