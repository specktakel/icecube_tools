import numpy as np
import os
from abc import ABC, abstractmethod
from itertools import product
from icecube_tools.utils.data import (
    IceCubeData,
    find_files,
    find_folders,
    data_directory,
    available_irf_periods,
    available_data_periods,
    RealEvents,
)
from scipy.interpolate import (
    RectBivariateSpline,
    RegularGridInterpolator,
    splrep,
    splev,
)
from scipy.integrate import quad

"""
Module for working with the public IceCube
effective area information.
"""

R2013_AEFF_FILENAME = "effective_areas"
R2015_AEFF_FILENAME = "effective_area.h5"
R2018_AEFF_FILENAME = "TabulatedAeff.txt"
BRAUN2008_AEFF_FILENAME = "AeffBraun2008.csv"
R2021_AEFF_FILENAME = "effectiveArea.csv"

_supported_dataset_ids = ["20131121", "20150820", "20181018", "20210126"]


class IceCubeAeffReader(ABC):
    """
    Abstract base class for a file reader to handle
    the different types of data files provided
    on the IceCube website.
    """

    def __init__(self, filename):
        """
        Abstract base class for a file reader to handle
        the different types of data files provided
        on the IceCube website.

        :param filename: name of the file to be read from (string).
        """

        self._filename = filename

        self.effective_area_values = None

        self.true_energy_bins = None

        self.cos_zenith_bins = None

        self._label_order = {"true_energy": 0, "cos_zenith": 1}

        self._units = {"effective_area": "m^2", "true_energy": "GeV", "cos_zenith": ""}

        self.read()

        super().__init__()

    @abstractmethod
    def read(self):
        """
        To be defined.
        """

        pass


class R2013AeffReader(IceCubeAeffReader):
    """
    Reader for the 2013 Nov 21 release.
    Link: https://icecube.wisc.edu/data-releases/2013/11/
    search-for-contained-neutrino-events-at-energies-above
    -30-tev-in-2-years-of-data.
    """

    def __init__(self, filename, **kwargs):
        """
        Here, filename is the path to the folder, as
        the effective areas are given in a bunch of
        separate files.
        """

        if "nu_flavors" in kwargs:

            self.nu_flavors = kwargs["nu_flavors"]

        else:

            self.nu_flavors = ["numu", "nue", "nutau"]

        if "scale_factor" in kwargs:

            self.scale_factor = kwargs["scale_factor"]

        else:

            self.scale_factor = 1

        self._cosz_range = np.linspace(-1, 1, 21)

        self._fname_str = "_cosZenRange_from_%+.1f_to_%+.1f.txt"

        super().__init__(filename)

    def read(self):

        self.cos_zenith_bins = self._cosz_range

        self.true_energy_bins = self._get_true_energy_bins()

        self.effective_area_values = np.zeros(
            (self.true_energy_bins.size - 1, self.cos_zenith_bins.size - 1)
        )

        for i, bounds in enumerate(
            zip(self.cos_zenith_bins[:-1], self.cos_zenith_bins[1:])
        ):

            l, u = bounds

            for nu_flavor in self.nu_flavors:

                tmp_file_name = nu_flavor + self._fname_str % (l, u)

                file_name = os.path.join(self._filename, tmp_file_name)

                tmp_read = np.loadtxt(file_name, skiprows=2).T

                self.effective_area_values.T[i] += tmp_read[2]

            # Assume equal flavour ratio in flux
            # self.effective_area_values.T[i] /= 3

        self.effective_area_values *= self.scale_factor

    def _get_true_energy_bins(self):
        """
        These are the same in all files, so can
        just read out once.
        """

        tmp_file_name = self.nu_flavors[0] + self._fname_str % (
            self._cosz_range[0],
            self._cosz_range[1],
        )

        file_name = os.path.join(self._filename, tmp_file_name)

        tmp_read = np.loadtxt(file_name, skiprows=2).T

        energy_bins = np.append(tmp_read[0], tmp_read[1][-1])

        return energy_bins


class R2015AeffReader(IceCubeAeffReader):
    """
    Reader for the 2015 Aug 20 release.
    Link: https://icecube.wisc.edu/science/data/HE_NuMu_diffuse.
    """

    def __init__(self, filename, **kwargs):

        if "year" in kwargs:
            self.year = kwargs["year"]
        else:
            self.year = 2011

        if "nu_type" in kwargs:
            self.nu_type = kwargs["nu_type"]
        else:
            self.nu_type = "nu_mu"

        if "scale_factor" in kwargs:

            self.scale_factor = kwargs["scale_factor"]

        else:

            self.scale_factor = 1

        super().__init__(filename)

        self._label_order["reco_energy"] = 2

        self._units["reco_energy"] = "GeV"

    def read(self):
        """
        Read input from the provided HDF5 file.
        """

        import h5py

        with h5py.File(self._filename, "r") as f:

            directory = f[str(self.year) + "/" + self.nu_type + "/"]

            self.effective_area_values = directory["area"][()]

            self.true_energy_bins = directory["bin_edges_0"][()]

            self.cos_zenith_bins = directory["bin_edges_1"][()]

            self.reco_energy_bins = directory["bin_edges_2"][()]

        self.effective_area_values *= self.scale_factor


class R2021AeffReader(IceCubeAeffReader):
    """
    Reader for the 2021 January 26 release.
    Link: https://icecube.wisc.edu/data-releases/2021/01/all-sky-point-source-icecube-data-years-2008-2018/
    """

    def __init__(self, filename, **kwargs):

        if "year" in kwargs:
            self.year = kwargs["year"]
        else:
            self.year = 2012

        # is this applicable? Simulation done using muon neutrino events
        if "nu_type" in kwargs:
            self.nu_type = kwargs["nu_type"]
        else:
            self.nu_type = "nu_mu"

        if "scale_factor" in kwargs:

            self.scale_factor = kwargs["scale_factor"]

        else:

            self.scale_factor = 1

        super().__init__(filename)

    def read(self):

        import pandas as pd
        from .r2021 import EMAX

        self.nu_type = "nu_mu"  # all entries valid for muons

        filelayout = ["Emin", "Emax", "DECmin", "DECmax", "Aeff"]
        # Aeff values are given in cm^2, multiply by 1e-4 to get m^2

        output = pd.read_csv(
            self._filename, comment="#", sep="\s+", names=filelayout
        ).to_dict()

        true_energy_lower = set(output["Emin"].values())
        true_energy_upper = set(output["Emax"].values())

        dec_lower = set(output["DECmin"].values())
        dec_upper = set(output["DECmax"].values())

        self.true_energy_bins = np.array(
            list(true_energy_upper.union(true_energy_lower))
        )
        self.true_energy_bins.sort()
        self.true_energy_bins = np.power(10, self.true_energy_bins)

        dec_bins = np.radians(np.array(list(dec_upper.union(dec_lower))))
        dec_bins.sort()
        self.cos_zenith_bins = np.cos(
            dec_bins + np.pi / 2
        )  # convert DEC to z and take cosine
        self.cos_zenith_bins.sort()  # sort to conform to existing data format

        self.effective_area_values = np.reshape(
            np.array(list(output["Aeff"].values())) * 1e-4,
            (len(dec_lower), len(true_energy_lower)),
        ).T
        self.effective_area_values = np.flip(
            self.effective_area_values, axis=1
        )  # flip due to sort, see above

        self.effective_area_values *= self.scale_factor

        # Need to mask out certain effective area values because there is no IRF defined for some
        true_energy_bins = np.power(10, np.arange(2.0, 9.1, 0.5))
        declination_bins = np.deg2rad(np.array([-90.0, -10.0, 10.0, 90.0]))

        mask = np.ones_like(self.effective_area_values)
        # Hardcoding this because I took long way too long trying to do it properly
        # Faulty IRF bins (etrue, dec) taken from the IRF class,
        # hardcoded because I couldn't resolve circular imports
        # Only take those bins where there is conflict to save some time upon loading
        faulty = {
            "IC40": [
                (-1, 0),
                (-1, 1),
                (-1, 2),
            ],  # aeff extends to higher energies than IRF
            "IC59": [(2, 0)],
        }
        self.mask = mask
        for period in faulty.keys():
            if period + "_" in os.path.basename(self._filename):

                # print(period)
                for etrue, dec in faulty[period]:
                    # print(etrue, dec)
                    if etrue != -1:
                        etrue_min, etrue_max = (
                            true_energy_bins[etrue],
                            true_energy_bins[etrue + 1],
                        )
                        aeff_etrue_min = (
                            np.digitize(etrue_min, self.true_energy_bins) - 1
                        )
                        aeff_etrue_max = (
                            np.digitize(etrue_max, self.true_energy_bins) - 1
                        )
                        aeff_etrue_max += (
                            1
                            if self.true_energy_bins[aeff_etrue_max]
                            < true_energy_bins[etrue + 1]
                            else 0
                        )
                    else:
                        etrue_min = true_energy_bins[-1]
                        aeff_etrue_min = (
                            np.digitize(etrue_min, self.true_energy_bins) - 1
                        )
                        aeff_etrue_max = self.true_energy_bins.size - 1
                    cosz_max, cosz_min = -np.sin(declination_bins[dec]), -np.sin(
                        declination_bins[dec + 1]
                    )

                    aeff_cosz_min = np.digitize(cosz_min, self.cos_zenith_bins) - 1
                    aeff_cosz_max = np.digitize(cosz_max, self.cos_zenith_bins) - 1
                    for e, c in product(
                        range(aeff_etrue_min, aeff_etrue_max),
                        range(aeff_cosz_min, aeff_cosz_max),
                    ):
                        # print(e, c)
                        self.mask[e, c] = 0.0
                self.effective_area_values = np.multiply(
                    self.effective_area_values, mask
                )
                break
        # Mask all entries which are beyond the IRF defined energy (relevant only beyond 1e9GeV)
        idx = np.digitize(EMAX, self.true_energy_bins) - 1
        self.effective_area_values = self.effective_area_values[:idx, :]
        self.true_energy_bins = self.true_energy_bins[: idx + 1]


class R2018AeffReader(IceCubeAeffReader):
    """
    Reader for the 2018 Oct 18 release.
    Link: https://icecube.wisc.edu/science/data/PS-3years.
    """

    def __init__(self, filename, **kwargs):

        if "scale_factor" in kwargs:

            self.scale_factor = kwargs["scale_factor"]

        else:

            self.scale_factor = 1

        super().__init__(filename)

    def read(self):

        import pandas as pd

        self.year = int(self._filename[-22:-18])
        self.nu_type = "nu_mu"

        filelayout = ["Emin", "Emax", "cos(z)min", "cos(z)max", "Aeff"]
        output = pd.read_csv(
            self._filename, comment="#", delim_whitespace=True, names=filelayout
        ).to_dict()

        true_energy_lower = set(output["Emin"].values())
        true_energy_upper = set(output["Emax"].values())

        cos_zenith_lower = set(output["cos(z)min"].values())
        cos_zenith_upper = set(output["cos(z)max"].values())

        self.true_energy_bins = np.array(
            list(true_energy_upper.union(true_energy_lower))
        )
        self.true_energy_bins.sort()

        self.cos_zenith_bins = np.array(list(cos_zenith_upper.union(cos_zenith_lower)))
        self.cos_zenith_bins.sort()

        self.effective_area_values = np.reshape(
            list(output["Aeff"].values()),
            (len(true_energy_lower), len(cos_zenith_lower)),
        )

        self.effective_area_values *= self.scale_factor


class Braun2008AeffReader(IceCubeAeffReader):
    """
    Reader for the Braun+2008 paper effective area
    Fig. 3, 140-150 deg.
    Braun, J. et al., 2008. Methods for point source
    analysis in high energy neutrino telescopes.
    Astroparticle Physics, 29(4), pp.299–305.
    """

    def read(self):

        self.nu_type = "nu_mu"

        out = np.loadtxt(self._filename, delimiter=",", comments="#")

        # Assume whole sky like this
        self.cos_zenith_bins = np.array([-1, 1])

        self.true_energy_bins = out.T[0]

        aeff = out.T[1]

        self.effective_area_values = aeff[:-1] + np.diff(aeff) / 2


class EffectiveArea(object):
    """
    IceCube effective area.
    """

    supported_datasets = _supported_dataset_ids

    def __init__(self, filename, **kwargs):
        """
        IceCube effective area.

        :param filename: name of the file to be read from (string).
        :param kwargs: kwargs to be passed to reader if relevant.
        """

        self._filename = filename

        self._reader = self.get_reader(**kwargs)

        self._spline_degree = kwargs.get("spline_degree", None)

        self._spline_smooth = kwargs.get("s", 0)

        self._interp = kwargs.get("interp", False)

        self.values = self._reader.effective_area_values.copy()

        self.true_energy_bins = self._reader.true_energy_bins

        self.cos_zenith_bins = self._reader.cos_zenith_bins

        self._integrate_out_ancillary_params()

        if isinstance(self._spline_degree, int):

            self._make_spline()

        elif self._interp:

            self._make_interp()

    def __call__(self, energy, cosz):
        """
        Return effective areas at given energy and cosz
        """

        if isinstance(energy, np.ndarray) and isinstance(cosz, np.ndarray):
            assert energy.shape == cosz.shape

        if hasattr(self, "_spline"):

            return self._spline_call(energy, cosz)

        elif self._interp:

            return self._interp_call(energy, cosz)

        energy_idx = np.digitize(energy, self.true_energy_bins) - 1

        cosz_idx = np.digitize(cosz, self.cos_zenith_bins) - 1

        return self.values[energy_idx, cosz_idx]

    def _histogam_call(self, energy, cosz):

        energy_idx = np.digitize(energy, self.true_energy_bins) - 1

        cosz_idx = np.digitize(cosz, self.cos_zenith_bins) - 1

        return self.values[energy_idx, cosz_idx]

    def _spline_call(self, energy, cosz):

        return self._spline(np.log10(energy), cosz, grid=False)

    def _make_interp(self):

        logE = np.log10(self.true_energy_bins)
        E = (logE[:-1] + logE[1:]) / 2
        c = (self.cos_zenith_bins[:-1] + self.cos_zenith_bins[1:]) / 2
        vals = self.values.copy()
        # mask out zero entries with 1e-10, in log -> -10 to avoid -infs
        # vals[vals==0] = 1e-10
        self._interpolation = RegularGridInterpolator(
            (E, c),
            vals,
            bounds_error=False,
            fill_value=0.0,
            method="cubic",
        )

    def _interp_call(self, energy, cosz):

        energy = np.log10(energy)

        if isinstance(energy, np.ndarray):
            args = np.vstack((energy, cosz)).T
        else:
            args = np.array([energy, cosz])

        return self._interpolation(args)

    def _make_spline(self):

        degree = self._spline_degree

        logE = np.log10(self.true_energy_bins)
        E = (logE[:-1] + logE[1:]) / 2
        c = (self.cos_zenith_bins[:-1] + self.cos_zenith_bins[1:]) / 2
        vals = self.values.copy()
        # mask out zero entries with 1e-10, in log -> -10 to avoid -infs
        # vals[vals==0] = 1e-10
        self._spline = RectBivariateSpline(E, c, vals, kx=degree, ky=degree, s=0)

    def get_reader(self, **kwargs):
        """
        Define an IceCubeAeffReader based on the filename.
        """

        if R2013_AEFF_FILENAME in self._filename:

            return R2013AeffReader(self._filename, **kwargs)

        elif R2015_AEFF_FILENAME in self._filename:

            return R2015AeffReader(self._filename, **kwargs)

        elif R2018_AEFF_FILENAME in self._filename:

            return R2018AeffReader(self._filename, **kwargs)

        elif BRAUN2008_AEFF_FILENAME in self._filename:

            return Braun2008AeffReader(self._filename)

        elif R2021_AEFF_FILENAME in self._filename:

            return R2021AeffReader(self._filename, **kwargs)

        else:

            raise ValueError(
                self._filename
                + " is not recognised as one of the known effective area files."
            )

    def _integrate_out_ancillary_params(self):
        """
        Sometimes the effective area is given as a
        function of ancillary parameters, e.g. the
        reconstructed muon energy. To give a unified
        interface, these can be integrated over.
        """

        if len(np.shape(self.values)) > 2:

            dim_to_int = []

            for key in self._reader._label_order:

                if "true_energy" not in key and "cos_zenith" not in key:

                    dim_to_int.append(self._reader._label_order[key])

            self.values = np.sum(self.values, axis=tuple(dim_to_int))

    def detection_probability(self, true_energy, true_cos_zenith, max_energy):
        """
        Give the relative detection probability for
        a given true energy and arrival direction.
        """

        # make copy of data array
        scaled_values = self.values.copy()
        # get lower edges of each bin, set prob to zero for all bins above inputted max energy
        lower_bin_edges = self.true_energy_bins[:-1]
        scaled_values[lower_bin_edges > max_energy] = 0
        # scale to max value: Aeff -> relative detection prob
        scaled_values = scaled_values / np.max(scaled_values)
        # find appropriate bin of inputted energies
        energy_index = np.digitize(true_energy, self.true_energy_bins) - 1

        # Guard against overflow
        if isinstance(energy_index, np.ndarray):
            idx = np.nonzero(energy_index >= self.true_energy_bins.size - 1)
            energy_index[idx] = self.true_energy_bins.size - 2

            idx = np.nonzero(energy_index < 0)
            energy_index[idx] = 0

            if len(self.cos_zenith_bins) > 2:
                cosz_index = np.digitize(true_cos_zenith, self.cos_zenith_bins) - 1
                return scaled_values[energy_index, cosz_index]
            else:
                return scaled_values[energy_index]

        else:
            if energy_index >= self.true_energy_bins.size - 1:

                energy_index = self.true_energy_bins.size - 2

            elif energy_index < 0:

                energy_index = 0
            if len(self.cos_zenith_bins) > 2:

                cosz_index = np.digitize(true_cos_zenith, self.cos_zenith_bins) - 1
                return scaled_values[energy_index][cosz_index]

            else:

                return scaled_values[energy_index]

    def get_splined_detection_probability(self, Elow, Ehigh, dec):
        # sindec = -cosz

        Elow = np.atleast_1d(Elow)
        Ehigh = np.atleast_1d(Ehigh)

        cosz_idx = np.digitize(-np.sin(dec), self.cos_zenith_bins) - 1
        aeff_slice = self.values[:, cosz_idx]

        dE = np.diff(self.true_energy_bins)

        dAeff_dE = aeff_slice / dE

        x = np.zeros(self.true_energy_bins.size + 1, dtype=np.double)
        x[0] = self.true_energy_bins[0]
        x[1:-1] = self.true_energy_bins[0] + np.diff(self.true_energy_bins) / 2
        x[-1] = self.true_energy_bins[-1]

        y = np.zeros(self.true_energy_bins.size + 1, dtype=np.double)
        y[0] = dAeff_dE[0]
        y[1:-1] = dAeff_dE
        y[-1] = dAeff_dE[-1]

        spline = splrep(x, y, k=1, s=0)

        def integral(Elow, Ehigh):
            # maybe better to use logspace as domain?
            return quad(splev, Elow, Ehigh, (spline, 0, 1), limit=200, full_output=1)[0]

        norm = integral(Elow.min(), Ehigh.max())
        probs = (
            np.array(
                [
                    integral(El, Eh)
                    for El, Eh in zip(
                        Elow,
                        Ehigh,
                    )
                ]
            )
            / norm
        )

        return probs

    @classmethod
    def from_dataset(cls, dataset_id, period="IC86_II", fetch=True, **kwargs):
        """
        Build effective area from a public dataset.

        If relevant, uses the latest effective area
        within a dataset. Some datasets have multiple
        effective areas.

        :param dataset_id: Date of dataset release e.g. 20181018
        :param period: For 20210126, which period should be used. Defaults to IC86_II (most recent)
        :param fetch: If true, download dataset if not existing
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

            files = find_files(dataset_dir, R2018_AEFF_FILENAME)

            # Latest dataset
            aeff_file_name = files[1]

        elif dataset_id == "20150820":

            files = find_files(dataset_dir, R2015_AEFF_FILENAME)

            # Latest dataset
            aeff_file_name = files[0]

        elif dataset_id == "20131121":

            folders = find_folders(dataset_dir, R2013_AEFF_FILENAME)

            # Folder containing all Aeff info
            aeff_file_name = folders[0]

        elif dataset_id == "20210126":
            if period not in available_irf_periods:
                raise ValueError(f"Period {period} is not supported.")

            files = find_files(dataset_dir, R2021_AEFF_FILENAME)
            for f in files:
                if "_".join((period, R2021_AEFF_FILENAME)) in f:
                    aeff_file_name = f
                    break
        return cls(aeff_file_name, period=period, **kwargs)
