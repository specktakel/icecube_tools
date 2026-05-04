from abc import ABC
from ast import Pass
from typing import Dict

from .effective_area import EffectiveArea
from .energy_resolution import EnergyResolution
from .angular_resolution import AngularResolution
from .r2021 import R2021IRF

"""
Detector modules, bringing together 
effective area and energy resolution.
"""


class Detector(ABC):
    """
    Abstract base class for a neutrino detector.
    """

    @property
    def period(self):

        return self._period

    @property
    def effective_area(self):

        return self._effective_area

    @effective_area.setter
    def effective_area(self, value):

        if not isinstance(value, EffectiveArea):

            raise ValueError(str(value) + " is not an instance of EffectiveArea")

        else:

            self._effective_area = value

    @property
    def energy_resolution(self):

        return self._energy_resolution

    @energy_resolution.setter
    def energy_resolution(self, value):

        if not isinstance(value, EnergyResolution):

            raise ValueError(str(value) + " is not an instance of EnergyResolution")

    @property
    def angular_resolution(self):

        return self._angular_resolution

    @angular_resolution.setter
    def angular_resolution(self, value):

        if not isinstance(value, AngularResolution):

            raise ValueError(str(value) + " is nmot an instance of AngularResolution.")

        else:

            self._angular_resolution = value


class IceCube(Detector):
    """
    IceCube detector.
    """

    def __init__(
        self, effective_area, energy_resolution, angular_resolution, period=None
    ):
        """
        IceCube detector.

        :param effective_area: instance of EffectiveArea.
        :param energy_resolution: instance of EnergyResolution.
        :param angular_resolution: instance of AngularResolution.
        """

        self._effective_area = effective_area

        self._energy_resolution = energy_resolution

        self._angular_resolution = angular_resolution

        self._period = period

        super().__init__()

    @classmethod
    def from_period(
        cls, period: str, release: str = "20210126"
    ):  # TODO change to new ID
        """
        Generate a detector from a period string of 2021 data release
        """

        aeff = EffectiveArea.from_dataset(release, period)
        irf = R2021IRF.from_period(release, period)

        return cls(aeff, irf, irf)


class TimeDependentDetector(ABC):

    @property
    def available_periods(self):
        return self._available_periods

    @property
    def detectors(self):
        return self._detectors

    @property
    def periods(self):
        return list(self._detectors.keys())

    def __getitem__(self, key):
        return self._detectors[key]


class TimeDependentIceCube(TimeDependentDetector):
    """
    Class to organise a time-dependent detector.
    """

    _available_periods = ["IC40", "IC59", "IC79", "IC86_I", "IC86_II"]

    def __init__(self, detectors):
        self._detectors = detectors

    @classmethod
    def from_periods(cls, *periods):
        """
        Creates class instance with a detector model for each given
        data taking period.
        :param periods: Tuple of strings, available ones listed above.
        :return:  `TimeDependentIceCube` instance.
        """

        # Check if all periods are supported
        if not all(_ in cls._available_periods for _ in periods):
            raise ValueError("Some period not supported.")

        # Empty dict to store detectors for all periods, key is period
        detectors = {}

        # Create detector instance for each period, return class instance
        for p in periods:
            aeff = EffectiveArea.from_dataset("20210126", p, fetch=False)
            irf = R2021IRF.from_period(p, fetch=False)
            detectors[p] = IceCube(aeff, irf, irf, p)
        return cls(detectors)

    def yield_detectors(self):
        for p, det in self.detectors.items():
            yield p, det
