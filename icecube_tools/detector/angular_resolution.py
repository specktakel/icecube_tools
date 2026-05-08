import numpy as np
from abc import ABC, abstractmethod
from vMF import sample_vMF
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy import stats
from scipy.stats import rv_histogram, uniform
from scipy.spatial.transform import Rotation as R
from icecube_tools.utils.data import IceCubeData, find_files, data_directory
from icecube_tools.utils.vMF import get_kappa, get_theta_p

"""
Module for handling the angular resolution
of IceCube based on public data.
"""

R2018_ANG_RES_FILENAME = "AngRes.txt"
R2015_ANG_RES_FILENAME = "angres_plot"

TRUE_ENERGY = 0
RECO_ENERGY = 1

_supported_dataset_ids = ["20181018"]


class IceCubeAngResReader(ABC):
    """
    Abstract base class for different input files
    from the IceCube website.
    """

    def __init__(self, filename):
        """
        Abstract base class for different input files
        from the IceCube website.

        :param filename: Name of file to read.
        """

        self._filename = filename

        self.ang_res_values = None

        self.true_energy_bins = None

        self.reco_energy_values = None

        self.prob_contained = None

        self.read()

        super().__init__()

    @abstractmethod
    def read(self):

        pass


class R2015AngResReader(IceCubeAngResReader):
    """
    Reader for the 2015 release.
    """

    def read(self):

        out = np.loadtxt(self._filename, delimiter=",", comments="#")

        self.ang_res_values = out.T[1]

        self.reco_energy_values = out.T[0]

        self.prob_contained = 0.68


class FromPlotAngResReader(IceCubeAngResReader):
    """
    Reader for the plots from the Aartsen+2018
    point source analysis paper.
    """

    def read(self):

        out = np.loadtxt(self._filename, delimiter=",", comments="#")

        self.ang_res_values = out.T[1]

        self.true_energy_values = out.T[0]

        self.true_energy_bins = None

        self.prob_contained = 0.68


class R2018AngResReader(IceCubeAngResReader):
    """
    Reader for the 2018 Oct 18 release.
    Link: https://icecube.wisc.edu/science/data/PS-3years.
    """

    def read(self):

        self.prob_contained = 0.68

        import pandas as pd

        self.year = int(self._filename[-15:-11])
        self.nu_type = "nu_mu"

        filelayout = ["E_min [GeV]", "E_max [GeV]", "Med. Resolution[deg]"]
        output = pd.read_csv(
            self._filename, comment="#", delim_whitespace=True, names=filelayout
        ).to_dict()

        true_energy_lower = set(output["E_min [GeV]"].values())
        true_energy_upper = set(output["E_max [GeV]"].values())

        self.true_energy_bins = np.array(
            list(true_energy_upper.union(true_energy_lower))
        )
        self.true_energy_bins.sort()

        self.ang_res_values = np.array(list(output["Med. Resolution[deg]"].values()))

        self.true_energy_values = (
            self.true_energy_bins[0:-1] + np.diff(self.true_energy_bins) / 2
        )


class AngularResolution(object):
    """
    Generic angular resolution class.
    """

    supported_datasets = _supported_dataset_ids

    def __init__(
        self,
        filename,
        ret_ang_err_p=0.68,
        offset=0,
        scale=1,
        scatter=None,
        minimum=0.1,
        maximum=10,
    ):
        """
        Generic angular resolution class.

        :param filename: File to load from
        :param ret_ang_err: Returned angular error will conrrespond to
        the radius of a region containing this probability
        :param offset: Add an offset to the read values.
        :param scale: Add a scale factor to the read values.
        :param scatter: Add scatter around read values.
        :param minimum: Specify minimum possible resolution
        :param maxmimum: Specify maximum possible resolution
        """

        self._filename = filename

        self._reader = self.get_reader()
        # put self.values somewhere else, not needed for every child class
        self.values = (self._reader.ang_res_values + offset) * scale

        if self._energy_type == TRUE_ENERGY:

            self.true_energy_bins = self._reader.true_energy_bins

            self.true_energy_values = self._reader.true_energy_values

        elif self._energy_type == RECO_ENERGY:

            self.reco_energy_values = self._reader.reco_energy_values

        self.ang_err_p = self._reader.prob_contained

        self.ret_ang_err_p = ret_ang_err_p

        self._ret_ang_err = None

        self._scatter = scatter

        self._minimum = minimum

        self._maximum = maximum

    def get_reader(self):
        """
        Define an IceCubeAeffReader based on the filename.
        """

        if R2018_ANG_RES_FILENAME in self._filename:

            self._energy_type = TRUE_ENERGY

            return R2018AngResReader(self._filename)

        elif R2015_ANG_RES_FILENAME in self._filename:

            self._energy_type = RECO_ENERGY

            return R2015AngResReader(self._filename)

        elif ".csv" in self._filename:

            self._energy_type = TRUE_ENERGY

            return FromPlotAngResReader(self._filename)

        else:

            raise ValueError(
                self._filename
                + " is not recognised as one of the known angular resolution files."
            )

    def _get_ang_err(self, E):
        """
        Get the median angular error for the
        given Etrue/Ereco, corresponding to prob_contained.

        If scatter, sample from a normal distribution
        centred on the median value.
        """

        # Get median value for this true energy
        if self._energy_type == TRUE_ENERGY:

            true_energy_bin_cen = (
                self.true_energy_bins[:-1] + self.true_energy_bins[1:]
            ) / 2

            ang_res = np.interp(np.log(E), np.log(true_energy_bin_cen), self.values)

        elif self._energy_type == RECO_ENERGY:

            ang_res = np.interp(np.log(E), np.log(self.reco_energy_values), self.values)

        # Add scatter if required
        if self._scatter:

            a = (self._minimum - ang_res) / self._scatter
            b = (self._maximum - ang_res) / self._scatter

            if isinstance(ang_res, np.ndarray):
                # shouldn't this be rvs(size=ang_res.size)?
                ang_res = stats.truncnorm(
                    a, b, loc=ang_res, scale=np.full(ang_res.shape, self._scatter)
                ).rvs()
            else:
                ang_res = stats.truncnorm(
                    a,
                    b,
                    loc=ang_res,
                    scale=self._scatter,
                ).rvs(
                    1
                )[0]

        # Check bounds
        if isinstance(ang_res, np.ndarray):
            idx = np.nonzero(ang_res < self._minimum)
            ang_res[idx] = self._minimum

            idx = np.nonzero(ang_res > self._maximum)
            ang_res[idx] = self._maximum

        else:
            if ang_res < self._minimum:

                ang_res = self._minimum

            if ang_res > self._maximum:

                ang_res = self._maximum

        return ang_res

    def get_ret_ang_err(self, E):
        """
        Get the median angular resolution for the
        given Etrue/Ereco, corresponsing to ret_ang_err_p.
        """

        ang_err = self._get_ang_err(E)

        kappa = get_kappa(ang_err, self.ang_err_p)

        return get_theta_p(kappa, self.ret_ang_err_p)

    def sample(self, Etrue, coord):
        """
        Sample new ra, dec values given a true energy
        and direction.
        """
        isarray = True
        ra, dec = coord

        if not isinstance(ra, np.ndarray):
            ra = np.array([ra])
            isarray = False
        if not isinstance(dec, np.ndarray):
            dec = np.array([dec])

        assert dec.shape == ra.shape

        if not isinstance(Etrue, np.ndarray):
            Etrue = np.array([Etrue])

        assert dec.shape == Etrue.shape

        ang_err = self._get_ang_err(Etrue)

        sky_coord = SkyCoord(ra=ra * u.rad, dec=dec * u.rad, frame="icrs")

        sky_coord.representation_type = "cartesian"

        unit_vector = np.array([sky_coord.x, sky_coord.y, sky_coord.z])
        kappa = get_kappa(ang_err, self.ang_err_p)
        new_unit_vector = sample_vMF(unit_vector, kappa)

        if new_unit_vector.shape != (3, Etrue.size):
            new_unit_vector = new_unit_vector.T

        new_sky_coord = SkyCoord(
            x=new_unit_vector[0],
            y=new_unit_vector[1],
            z=new_unit_vector[2],
            representation_type="cartesian",
        )

        new_sky_coord.representation_type = "unitspherical"

        new_ra = new_sky_coord.ra.rad

        new_dec = new_sky_coord.dec.rad

        self._ret_ang_err = get_theta_p(kappa, self.ret_ang_err_p)

        if not isarray:
            self._ret_ang_err = self._ret_ang_err[0]

        return new_ra, new_dec

    @property
    def ret_ang_err(self):

        return self._ret_ang_err

    @classmethod
    def from_dataset(cls, dataset_id, fetch=True, **kwargs):
        """
        Load angular resolution from publicly
        available data.

        :dataset_id: ID date of the dataset e.g. "20181018"
        :param fetch: If true, download dataset if missing
        """

        if dataset_id not in _supported_dataset_ids:

            raise NotImplementedError("This dataset is not currently supported")

        if fetch:

            data_interface = IceCubeData()

            dataset = data_interface.find(dataset_id)

            data_interface.fetch(dataset)

            dataset_dir = data_interface.get_path_to(dataset[0])

        else:

            dataset_dir = data_directory

        if dataset_id == "20181018":

            files = find_files(dataset_dir, R2018_ANG_RES_FILENAME)

            angres_file_name = files[2]
        """
        elif dataset_id == "20210126":

            files = find_files(dataset_dir, R2021_ANG_RES_FILENAME)
            angres_file_name = files[0]
            return R2021AngularResolution(angres_file_name, **kwargs)
        """
        return AngularResolution(angres_file_name, **kwargs)


class FixedAngularResolution:
    """
    Simple fixed angular resolution.
    """

    def __init__(self, ang_err=1.0, ang_err_p=0.68, ret_ang_err_p=0.68):
        """
        Simple fixed angular resolution.

        :param ang_err: Resolution [deg].
        :param ang_err_p: The probability contained in the passed
        angular error region
        :param ret_ang_err_p: The returned angular error will correspond
        to a region containing this probability
        """

        self.ang_err = ang_err

        self.ang_err_p = ang_err_p

        self.ret_ang_err_p

        self._kappa = get_kappa(ang_err, ang_err_p)

        self._ret_ang_err = get_theta_p(self._kappa, ret_ang_err_p)

    @property
    def ret_ang_err(self):

        return self._ret_ang_err

    @property
    def kappa(self):

        return self._kappa

    def sample(self, coord):
        """
        Sample reconstructed coord given original position.

        :coord: ra, dec in [rad].
        """

        ra, dec = coord

        sky_coord = SkyCoord(ra=ra * u.rad, dec=dec * u.rad, frame="icrs")

        sky_coord.representation_type = "cartesian"

        unit_vector = np.array([sky_coord.x, sky_coord.y, sky_coord.z])

        new_unit_vector = sample_vMF(unit_vector, self._kappa, 1)[0]

        new_sky_coord = SkyCoord(
            x=new_unit_vector[0],
            y=new_unit_vector[1],
            z=new_unit_vector[2],
            representation_type="cartesian",
        )

        new_sky_coord.representation_type = "unitspherical"

        new_ra = new_sky_coord.ra.rad

        new_dec = new_sky_coord.dec.rad

        return new_ra, new_dec


def icrs_to_unit_vector(ra, dec):
    """
    Convert to unit vector.
    """

    theta = np.pi / 2 - dec
    phi = ra

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.array([x, y, z])


def unit_vector_to_icrs(unit_vector):
    """
    Convert to ra, dec.
    """

    x = unit_vector[0]
    y = unit_vector[1]
    z = unit_vector[2]

    phi = np.arctan(y / x)
    theta = np.arccos(z)

    ra = phi
    dec = np.pi / 2 - theta

    return ra, dec
