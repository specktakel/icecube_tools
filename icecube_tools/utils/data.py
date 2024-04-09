from re import A
import numpy as np
import os
from os.path import join
import requests
import requests_cache
import time
import tarfile
from zipfile import ZipFile
from bs4 import BeautifulSoup
from tqdm import tqdm
from astropy import units as u
import h5py
import logging
from abc import ABC, abstractmethod

from ..source.flux_model import BrokenPowerLawFlux, PowerLawFlux
from ..source.source_model import Source, DIFFUSE, POINT
from ..utils.vMF import get_kappa, get_theta_p

from typing import List, Dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

icecube_data_base_url = "https://icecube.wisc.edu/data-releases"
data_directory = os.path.abspath(os.path.join(os.path.expanduser("~"), ".icecube_data"))

available_irf_periods = ["IC40", "IC59", "IC79", "IC86_I", "IC86_II"]

available_data_periods = [
    "IC40",
    "IC59",
    "IC79",
    "IC86_I",
    "IC86_II",
    "IC86_III",
    "IC86_IV",
    "IC86_V",
    "IC86_VI",
    "IC86_VII",
]


class IceCubeData:
    """
    Handle the interface with IceCube's public data
    releases hosted on their website.
    """

    def __init__(
        self,
        base_url=icecube_data_base_url,
        data_directory=data_directory,
        cache_name=".cache",
        update=False,
    ):
        """
        Handle the interface with IceCube's public data
        releases hosted on their website.

        :param base_url: Base url for data releases
        :param data_directory: Where to put the data
        :param cache_name: Name of the requests cache
        :param update: Refresh the cache if true
        """

        self.base_url = base_url

        self.data_directory = data_directory

        requests_cache.install_cache(
            cache_name=cache_name,
            expire_after=-1,
        )

        self.ls(verbose=False, update=update)

        # Make data directory if it doesn't exist
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)

    def ls(self, verbose=True, update=False):
        """
        List the available datasets.

        :param verbose: Print the datasets if true
        :param update: Refresh the cache if true
        """

        self.datasets = []

        if update:
            requests_cache.clear()

        response = requests.get(self.base_url)

        if response.ok:
            soup = BeautifulSoup(response.content, "html.parser")

            links = soup.find_all("a")

            for link in links:
                href = link.get("href")

                if ".zip" in href:
                    self.datasets.append(href)

                    if verbose:
                        print(href)

    def find(self, search_string):
        """
        Find datasets containing search_string.
        """

        found_datasets = []

        for dataset in self.datasets:
            if search_string in dataset:
                found_datasets.append(dataset)

        return found_datasets

    def fetch(self, datasets, overwrite=False, write_to=None):
        """
        Downloads and unzips the given datasets.

        :param datasets: A list of dataset names
        :param overwrite: Overwrite existing files
        :param write_to: Optional custom location
        """

        if write_to:
            old_dir = self.data_directory

            self.data_directory = write_to

        for dataset in datasets:
            if dataset not in self.datasets:
                raise ValueError(
                    "Dataset %s is not in list of known datasets" % dataset
                )

            url = os.path.join(self.base_url, dataset)

            local_path = os.path.join(self.data_directory, dataset)

            # Only fetch if not already there!
            if not os.path.exists(os.path.splitext(local_path)[0]) or overwrite:
                # Don't cache this as we want to stream
                with requests_cache.disabled():
                    response = requests.get(url, stream=True)

                    if response.ok:
                        total = int(response.headers["content-length"])

                        # For progress bar description
                        short_name = dataset
                        if len(dataset) > 40:
                            short_name = dataset[0:40] + "..."

                        # Save locally
                        with open(local_path, "wb") as f, tqdm(
                            desc=short_name, total=total
                        ) as bar:
                            for chunk in response.iter_content(chunk_size=1024 * 1024):
                                size = f.write(chunk)
                                bar.update(size)

                        # Unzip
                        dataset_dir = os.path.splitext(local_path)[0]
                        with ZipFile(local_path, "r") as zip_ref:
                            zip_ref.extractall(dataset_dir)

                        # Delete zipfile
                        os.remove(local_path)

                        # Check for further compressed files in the extraction
                        tar_files = find_files(dataset_dir, ".tar")

                        zip_files = find_files(dataset_dir, ".zip")

                        for tf in tar_files:
                            tar = tarfile.open(tf)
                            tar.extractall(os.path.splitext(tf)[0])

                        for zf in zip_files:
                            with ZipFile(zf, "r") as zip_ref:
                                zip_ref.extractall(zf[:-4])

                crawl_delay()

        if write_to:
            self.data_directory = old_dir

    def fetch_all_to(self, write_to, overwrite=False):
        """
        Download all data to a given location
        """

        self.fetch(self.datasets, write_to=write_to, overwrite=overwrite)

    def get_path_to(self, dataset):
        """
        Get path to a given dataset
        """

        if dataset not in self.datasets:
            raise ValueError("Dataset is not available")

        local_zip_loc = os.path.join(self.data_directory, dataset)

        local_path = os.path.splitext(local_zip_loc)[0]

        return local_path


class ddict(dict):
    """
    Modified dictionary class, derived from `dict`.
    Used to nest dictionaries without having to write [] all the time when adding or calling.
    """

    def __init__(self):
        super().__init__()

    def add(self, value, *keys):
        """
        Add value to chain of keys.
        Careful: this may overwrite existing values!
        :param value: Value to be added
        :param keys: Tuple containing ordered keys behind which the value should be added.
        """
        # TODO: protect from overwriting

        temp = self
        for key in keys[:-1]:
            try:
                temp[key]
            except KeyError:
                temp[key] = {}
            finally:
                temp = temp[key]
        temp[keys[-1]] = value

    def __call__(self, *keys):
        """
        Call value of nested dicts.
        :param keys: Tuple of ordered keys whose value should be returned
        :return: value behind tuple of keys
        """

        temp = self
        for key in keys:
            temp = temp[key]
        return temp


class Uptime:
    _available_data_periods = available_data_periods
    _available_irf_periods = available_irf_periods

    def __init__(self, *periods, **kwargs):
        try:
            np.loadtxt(
                os.path.join(
                    data_directory,
                    "20210126_PS-IC40-IC86_VII",
                    "icecube_10year_ps",
                    "uptime",
                    "IC40_exp.csv",
                )
            )
        except FileNotFoundError:
            data_interface = IceCubeData()
            dataset = data_interface.find("20210126")
            data_interface.fetch(dataset)

        self._irf_periods = []
        self._data_periods = []
        for p in periods:
            self._data_periods.append(p)
            if p in self._available_irf_periods:
                self._irf_periods.append(p)

        self._data = {}
        # Store start and end times of each period separately
        self._times = np.zeros((len(self._available_data_periods), 2))
        for c, p in enumerate(self._available_data_periods):
            self._data[p] = np.loadtxt(
                os.path.join(
                    data_directory,
                    "20210126_PS-IC40-IC86_VII",
                    "icecube_10year_ps",
                    "uptime",
                    f"{p}_exp.csv",
                )
            )
            self._times[c, 0] = self._data[p][0, 0]
            self._times[c, 1] = self._data[p][-1, -1]

    def time_span(self, IRF: bool = True):
        """
        Return dictionary (keys are data or irf periods) with good time intervals of observations.
        :param periods: Strings of data/irf periods.
        :param IRF: bool, if true observation time of IC86_III and later is counted towards IC86_II.
        :return: Dict of total times in years (without unit) between start and end of each queried data period.
        """

        output = {}
        for p in self._data_periods:
            time = self._time_span(p)
            if IRF and p in available_data_periods and not p in available_irf_periods:
                try:
                    output["IC86_II"] += time
                except KeyError:
                    output["IC86_II"] = time
            else:
                output[p] = time
        return output

    def _time_span(self, period: str):
        time = self._data[period][-1, -1] - self._data[period][0, 0]
        time = time * u.d
        return time.to("year").value

    def cumulative_time_obs(self, IRF: bool = True):
        """
        :param periods: Strings of data periods.
        :return: Return total observation time of each queried data period in years (without unit)
        """

        output = {}
        for p in self._data_periods:
            time = self._time_obs(p)
            if IRF and p in available_data_periods and p not in available_irf_periods:
                try:
                    output["IC86_II"] += time
                except KeyError:
                    output["IC86_II"] = time
            else:
                output[p] = time
        return output

    def _time_obs(self, period: str):
        intervals = self._data[period][:, 1] - self._data[period][:, 0]
        time = np.sum(intervals) * u.d
        return time.to("year").value

    def find_obs_time(self, IRF: bool = True, **kwargs):
        """
        Calculate the amounts of time in each period covered for either:
         - given start and end time (should be MJD)
         - duration and end date
         - duration and start date
        Duration should be in float in years.
        """

        start = kwargs.get("start", False)
        end = kwargs.get("end", False)
        duration = kwargs.get("duration", False)

        if start and end and not duration:
            duration = (end - start) * u.day
            duration = duration.to("year")
        elif start and duration and not end:
            duration = duration * u.year
            duration = duration.to("day")
            end = start + duration.value
            duration = duration.to("year")
        elif end and duration and not start:
            duration = duration * u.year
            duration = duration.to("day")
            start = end - duration.value
            duration = duration.to("year")
        else:
            raise ValueError("Not a supported combination of arguments.")

        if start < self._times[0, 0]:
            logger.warning(
                "Start time outside of running experiment, setting to earliest possible time."
            )
            start = self._times[0, 0]

        p_start = np.digitize(start, self._times[:, 0]) - 1

        if end >= self._times[-1, -1]:
            logger.info(
                "End time outside of provided data set, sending an owl to Professor Trelawney"
            )
            # Set to highest allowed value
            p_end = len(available_data_periods) - 1
            future = True

        else:
            # Do not subtract one because we need the entry after the digitized one
            p_end = np.digitize(end, self._times[:, 1])
            future = False

        # repeat searchsorted procedure for the periods containing start/end:
        # add up all the detector uptime in those to get the resulting obs time

        obs_times = {}
        if p_start == p_end and not future:
            fraction = duration / self._time_span(available_data_periods[p_start])
            t_obs = fraction * self._time_obs(available_data_periods[p_start])
            obs_times[available_data_periods[p_start]] = t_obs.value
        else:
            # find duration in start period:
            duration = ((self._times[p_start, 1] - start) * u.day).to("year")
            fraction = duration / self._time_span(available_data_periods[p_start])
            t_obs_start = fraction * self._time_obs(available_data_periods[p_start])
            obs_times[available_data_periods[p_start]] = t_obs_start.value

            # now for the middle periods:
            for c_p in range(p_start + 1, p_end):
                obs_times[available_data_periods[c_p]] = self._time_obs(
                    available_data_periods[c_p]
                )

            # end
            duration = ((end - self._times[p_end, 0]) * u.day).to("year")
            fraction = duration / self._time_span(available_data_periods[p_end])
            t_obs_end = fraction * self._time_obs(available_data_periods[p_end])
            obs_times[available_data_periods[p_end]] = t_obs_end.value

        if IRF:
            new_obs_times = {}
            for p, t in obs_times.items():
                if p in available_data_periods and p not in available_irf_periods:
                    try:
                        new_obs_times["IC86_II"] += t
                    except KeyError:
                        new_obs_times["IC86_II"] = t
                else:
                    new_obs_times[p] = t
            return new_obs_times
        else:
            return obs_times

    @property
    def irf_periods(self):
        return self._irf_periods

    @property
    def data_periods(self):
        return self._data_periods


class dddict(dict):
    def __init__(self):
        super().__init__()

    def min(self):
        for key, value in self.items():
            try:
                if value < minimum:
                    minimum = value
            except NameError:
                minimum = value
        return minimum

    def max(self):
        for key, value in self.items():
            try:
                if value > maximum:
                    maximum = value
            except NameError:
                maximum = value
        return maximum


class Events(ABC):
    """
    Abstract bass class for event classes
    For single period event files, the properties return not a dictionary but a single array of data.
    """

    def __init__(self, seed: int = 1234):
        self._create_dicts()
        self._periods = []
        self.mask = None
        self.seed = seed

    def _create_dicts(self):
        self._reco_energy = dddict()
        self._ang_err = dddict()
        self._ra = dddict()
        self._dec = dddict()

    def __len__(self):
        return len(self._periods)

    @abstractmethod
    def period(self):
        pass

    @classmethod
    @abstractmethod
    def load_from_h5(cls):
        pass

    @abstractmethod
    def write_to_h5(self):
        pass

    @abstractmethod
    def scramble_ra(self):
        pass

    @property
    def periods(self):
        return self._irf_periods

    @property
    def irf_periods(self):
        return self._irf_periods

    @property
    def data_periods(self):
        return self._data_periods

    @property
    def reco_energy(self):
        if self.mask:
            return {p: self._reco_energy[p][self.mask[p]] for p in self.periods}
        else:
            return self._reco_energy

    @property
    def ra(self):
        if self.mask:
            return {p: self._ra[p][self.mask[p]] for p in self.periods}
        else:
            return self._ra

    @property
    def dec(self):
        if self.mask:
            return {p: self._dec[p][self.mask[p]] for p in self.periods}
        else:
            return self._dec

    @property
    def ang_err(self):
        if self.mask:
            return {p: self._ang_err[p][self.mask[p]] for p in self.periods}
        else:
            return self._ang_err

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, s: int):
        self._seed = s
        self.rng = np.random.default_rng(seed=s)

    def _return_single_period(self, data):
        return data[self._periods[0]]

    def restrict(
        self,
        dec_low: float = -np.pi / 2,
        dec_high: float = np.pi / 2,
        ra_low=0.0,
        ra_high=2 * np.pi,
        ereco_low: float = 1.0,
    ):
        """
        Restrict declination to given range, if None provided respective bound is ignored.
        For fancier restrictions use `self.mask`.
        :param dec_low: Lower declination bound, in rad.
        :param dec_high: Upper declination bound, in rad.
        """

        if dec_low >= dec_high:
            raise ValueError("dec_low is greater than or equal to dec_high!")

        mask = {}
        for p in self.periods:
            mask[p] = np.nonzero(
                (
                    (self._dec[p] >= dec_low)
                    & (self._dec[p] <= dec_high)
                    & (self._reco_energy[p] >= ereco_low)
                    & (self._ra[p] >= ra_low)
                    & (self._ra[p] <= ra_high)
                )
            )
        self.mask = mask

    @property
    def N(self) -> Dict:
        output = {}
        for p in self.periods:
            output[p] = self._ra[p].size
        return output

    @property
    def N_restricted(self) -> Dict:
        output = {}
        ra = self.ra
        for p in self.periods:
            output[p] = ra[p].size
        return output


class SimEvents(Events):
    """
    Class to store simulated events.
    """

    ic86_ii_data_periods = [
        "IC86_II",
        "IC86_III",
        "IC86_IV",
        "IC86_V",
        "IC86_VI",
        "IC86_VII",
    ]

    keys = [
        "true_energy",
        "arrival_energy",
        "reco_energy",
        "ra",
        "dec",
        "ang_err",
        "source_label",
    ]

    def __init__(self, seed: int = 1234):
        super().__init__(seed=seed)
        self._true_energy = dddict()
        self._arrival_energy = dddict()
        self._source_label = {}

    @classmethod
    def load_from_h5(cls, path: str, seed: int = 1234):
        """
        Load events from hdf5 file.
        :param path: Path to file
        :param seed: Random seed for RA scrambling
        """

        inst = cls(seed=seed)
        inst._periods = []
        inst.path = path
        inst.sources = {}
        with h5py.File(inst.path, "r") as f:
            # TODO load source data too!
            for p, data in f.items():
                if not "source" in p:
                    inst._periods.append(p)
                    inst._true_energy[p] = data["true_energy"][()]
                    inst._reco_energy[p] = data["reco_energy"][()]
                    inst._arrival_energy[p] = data["arrival_energy"][()]
                    inst._ra[p] = data["ra"][()]
                    inst._dec[p] = data["dec"][()]
                    inst._ang_err[p] = data["ang_err"][()]
                    inst._source_label[p] = data["source_label"][()]
                else:
                    indices = []
                    for s in ["index", "index1", "index2"]:
                        try:
                            indices.append(data[s][()])
                        except:
                            pass
                    inst.sources[p] = indices
                inst._irf_periods = list(inst._reco_energy.keys())
                inst._data_periods = list(inst._reco_energy.keys())
        return inst

    def write_to_h5(self, path: str, sources: List[Source]):
        """
        Write events to hdf5 file.
        :param path: Path to file
        :sources: List of sources used to generate events
        """

        with h5py.File(path, "w") as f:
            for p in self._periods:
                group = f.create_group(p)
                group.create_dataset("true_energy", data=self._true_energy[p])
                group.create_dataset("arrival_energy", data=self._arrival_energy[p])
                group.create_dataset("reco_energy", data=self._reco_energy[p])
                group.create_dataset("ang_err", data=self._ang_err[p])
                group.create_dataset("ra", data=self._ra[p])
                group.create_dataset("dec", data=self._dec[p])
                group.create_dataset("source_label", data=self._source_label[p])

            for i, source in enumerate(sources):
                s = f.create_group("source_{}".format(str(i)))
                if isinstance(source.flux_model, PowerLawFlux):
                    s.create_dataset("index", data=source.flux_model._index)
                    s.create_dataset(
                        "normalisation_energy",
                        data=source.flux_model._normalisation_energy,
                    )
                elif isinstance(source.flux_model, BrokenPowerLawFlux):
                    s.create_dataset("index1", data=source.flux_model._index1)
                    s.create_dataset("index2", data=source.flux_model._index2)
                    s.create_dataset(
                        "break_energy", data=source.flux_model._break_energy
                    )
                s.create_dataset("source_type", data=source.source_type)
                s.create_dataset("normalisation", data=source.flux_model._normalisation)
                if source.source_type == POINT:
                    s.create_dataset("source_coord", data=source.coord)

    def period(self, p: str):
        """
        Returns dictionary of events belonging to a specified data season
        :param p: Period identifier
        """

        out = {}
        if self.mask:
            out["true_energy"] = self._true_energy[p][self.mask[p]]
            out["reco_energy"] = self._reco_energy[p][self.mask[p]]
            out["arrival_energy"] = self._arrival_energy[p][self.mask[p]]
            out["ang_err"] = self._ang_err[p][self.mask[p]]
            out["ra"] = self._ra[p][self.mask[p]]
            out["dec"] = self._dec[p][self.mask[p]]
            out["source_label"] = self._source_label[p][self.mask[p]]
        else:
            out["true_energy"] = self._true_energy[p]
            out["reco_energy"] = self._reco_energy[p]
            out["arrival_energy"] = self._arrival_energy[p]
            out["ang_err"] = self._ang_err[p]
            out["ra"] = self._ra[p]
            out["dec"] = self._dec[p]
            out["source_label"] = self._source_label[p]
        return out

    def scramble_ra(self):
        """
        Srcambles RA of all events, no distinction between source and background is made.
        If a mask is supplied, only unmasked events are scrambled to save time on sampling random numbers.
        """
        num_of_events = 0
        logger.info("Scrambling RAs.")
        for p in self.periods:
            num_of_events += self._ra[p].size
            if self.mask:
                self._ra[p][self.mask[p]] = self.rng.uniform(
                    low=0.0, high=2 * np.pi, size=self.mask[p].size
                )
            else:
                self._ra[p] = self.rng.uniform(
                    low=0.0, high=2 * np.pi, size=self._ra[p].size
                )

        if num_of_events < 10000:
            logger.warning(
                f"Shuffling RA with low ({num_of_events} events) statistics. Proceed with caution."
            )

    @property
    def true_energy(self):
        if self.mask:
            return {p: self._true_energy[p][self.mask[p]] for p in self.periods}
        else:
            return self._true_energy

    @property
    def arrival_energy(self):
        if self.mask:
            return {p: self._arrival_energy[p][self.mask[p]] for p in self.periods}
        else:
            return self._arrival_energy

    @property
    def source_label(self):
        if self.mask:
            return {p: self._source_label[p][self.mask[p]] for p in self.periods}
        else:
            return self._source_label


class RealEvents(Events):
    """
    Class to handle reading real event files of 2021 release.
    """

    mjd_ = 0
    reco_energy_ = 1
    ang_err_ = 2
    ra_ = 3
    dec_ = 4

    STACK = {}

    keys = ["reco_energy", "ra", "dec", "ang_err", "mjd"]

    def __init__(self, *data_periods, seed=1234):
        super().__init__(*data_periods, seed=seed)
        self._mjd = {}

    def period(self, p: str):
        """
        Returns dictionary of events belonging to a specified data season
        :param p: Period identifier
        """

        out = {}
        if self.mask:
            out["reco_energy"] = self._reco_energy[p][self.mask[p]]
            out["ang_err"] = self._ang_err[p][self.mask[p]]
            out["ra"] = self._ra[p][self.mask[p]]
            out["dec"] = self._dec[p][self.mask[p]]
            out["mjd"] = self._mjd[p][self.mask[p]]
        else:
            out["reco_energy"] = self._reco_energy[p]
            out["ang_err"] = self._ang_err[p]
            out["ra"] = self._ra[p]
            out["dec"] = self._dec[p]
            out["mjd"] = self._mjd[p]
        return out

    def _sort(self):
        """
        Sort event information in dictionaries by season,
        converts (assumed) 50% containment for angular errors to 68%,
        converts ra/dec from degrees to radians,
        converts log10(E/GeV) in GeV.
        """

        for p in self._periods:
            self._reco_energy[p] = np.power(10, self.events[p][:, self.reco_energy_])
            self._ang_err[p] = self.events[p][:, self.ang_err_]
            self._ra[p] = np.deg2rad(self.events[p][:, self.ra_])
            self._dec[p] = np.deg2rad(self.events[p][:, self.dec_])
            self._mjd[p] = self.events[p][:, self.mjd_]

    def _add(self):
        """
        Add events for multiple data seasons of a single IRF, i.e. only IC86_II and up.
        """

        to_be_added = []

        for p in self._periods:
            if p in ["IC86_II", "IC86_III", "IC86_IV", "IC86_V", "IC86_VI", "IC86_VII"]:
                to_be_added.append(p)

        reco_energy = [self._reco_energy[_] for _ in to_be_added]
        mjd = [self._mjd[_] for _ in to_be_added]
        ang_err = [self._ang_err[_] for _ in to_be_added]
        ra = [self._ra[_] for _ in to_be_added]
        dec = [self._dec[_] for _ in to_be_added]

        for p in to_be_added:
            del self._mjd[p]
            del self._ang_err[p]
            del self._reco_energy[p]
            del self._ra[p]
            del self._dec[p]
        if to_be_added:
            self._ra["IC86_II"] = np.concatenate(ra)
            self._dec["IC86_II"] = np.concatenate(dec)
            self._ang_err["IC86_II"] = np.concatenate(ang_err)
            self._reco_energy["IC86_II"] = np.concatenate(reco_energy)
            self._mjd["IC86_II"] = np.concatenate(mjd)

    def _copy_data(self, other: Events, p: str):
        self._mjd[p] = other._mjd[p].copy()
        self._ang_err[p] = other._ang_err[p].copy()
        self._reco_energy[p] = other._reco_energy[p].copy()
        self._ra[p] = other._ra[p].copy()
        self._dec[p] = other._dec[p].copy()

    @classmethod
    def from_event_files(
        cls, *periods: str, seed: int = 1234, use_all: bool = False, **kwargs
    ):
        """
        Load from files provided by data release,
        if belonging to IC86_II or later, add to IC86_II keyword
        because the same IRF is used
        :param periods: Arbitrary number of period identifiers
        :param seed: int, seed for rng
        :param use_all: bool, true if IC86_II should be translated to all data periods with this detector configuration
        """

        try:
            np.loadtxt(
                os.path.join(
                    data_directory,
                    "20210126_PS-IC40-IC86_VII",
                    "icecube_10year_ps",
                    "uptime",
                    "IC40_exp.csv",
                )
            )
        except FileNotFoundError:
            data_interface = IceCubeData()
            dataset = data_interface.find("20210126")
            data_interface.fetch(dataset)
            dataset_dir = data_interface.get_path_to(dataset[0])
        periods = list(periods)
        if not periods:
            # use all periods if none are specified
            periods = [
                "IC40",
                "IC59",
                "IC79",
                "IC86_I",
                "IC86_II",
                "IC86_III",
                "IC86_IV",
                "IC86_V",
                "IC86_VI",
                "IC86_VII",
            ]

        if use_all and "IC86_II" in periods:
            periods += [
                "IC86_II",
                "IC86_III",
                "IC86_IV",
                "IC86_V",
                "IC86_VI",
                "IC86_VII",
            ]

        # Get rid of possible double entries
        periods = list(set(periods))

        # Check in STACK if there is a singular period event instance
        # if not, create one from scratch
        # Then, copy data into a new instance encompassing all periods
        inst = cls(seed=seed)
        for p in available_data_periods:
            if not p in periods:
                continue
            if p in RealEvents.STACK:
                temp = RealEvents.STACK[p]
            else:
                temp = cls(seed=42)
                temp.events = {}
                temp.events[p] = np.loadtxt(
                    join(
                        data_directory,
                        f"20210126_PS-IC40-IC86_VII/icecube_10year_ps/events/{p}_exp.csv",
                    )
                )
                temp._periods.append(p)
                temp._sort()
                RealEvents.STACK[p] = temp
            inst._periods.append(p)
            # Copy stuff
            inst._copy_data(temp, p)

        # Compress IC86_II and onwards into a single entry
        inst._add()
        inst._uptime = Uptime(*inst._periods)
        inst._data_periods = inst._uptime.data_periods
        inst._irf_periods = inst._uptime.irf_periods
        return inst

    @classmethod
    def load_from_h5(cls, path: str):
        """
        Load events from hdf5 file
        :param path: Path to file
        """

        inst = cls()
        inst._periods = []
        with h5py.File(path, "r") as f:
            for p, data in f.items():
                inst._periods.append(p)
                inst._reco_energy[p] = data["reco_energy"][()]
                inst._ang_err[p] = data["ang_err"][()]
                inst._ra[p] = data["ra"][()]
                inst._dec[p] = data["dec"][()]
                inst._mjd[p] = data["time"][()]
        return inst

    def write_to_h5(self, path):
        """
        Write selected events to hdf5 file.
        :param path: Path to hdf5 file
        """

        with h5py.File(path, "w") as f:
            for p in self._periods:
                group = f.create_group(p)
                group.create_dataset("reco_energy", data=self._reco_energy[p])
                group.create_dataset("ang_err", data=self._ang_err[p])
                group.create_dataset("ra", data=self._ra[p])
                group.create_dataset("dec", data=self._dec[p])
                group.create_dataset("mjd", data=self._time[p])

    @property
    def mjd(self):
        if self.mask:
            return {p: self._mjd[p][self.mask[p]] for p in self.periods}
        else:
            return self._mjd

    def insert_fake_data(self, data: SimEvents):
        """
        Inserts fake data into real data for e.g. sensitivity testing.
        Should have some way of distinguishing between fake and real data?!
        Tested, works!
        :param data: :class:`icecube_tools.utils.data.SimEvents`
        """

        for p in data.periods:
            self._reco_energy[p] = np.hstack(
                (self._reco_energy[p], data._reco_energy[p])
            )
            self._ra[p] = np.hstack((self._ra[p], data._ra[p]))
            self._dec[p] = np.hstack((self._dec[p], data._dec[p]))
            self._ang_err[p] = np.hstack((self._ang_err[p], data._ang_err[p]))

    def scramble_ra(self):
        """
        Scrambles the RA of all events that are not inserted fakes. Done by using the size of MJD,
        which is not provided for fake data.
        Fake data is excluded s.t. inserted point sources are not accidentally washed out.
        #TODO: Restricting to unmasked events
        """
        num_of_events = 0
        logger.info("Scrambling RAs.")
        for p in self.periods:
            self._ra[p] = self.rng.uniform(
                low=0.0, high=2 * np.pi, size=self._mjd[p].size
            )
            # self._ra[p] = np.hstack((
            #    self.rng.uniform(low=0., high=2*np.pi, size=self._mjd[p].size),
            #    self._ra[p][self._mjd[p].size:]
            #    ))
            num_of_events += self._mjd[p].size
        if num_of_events < 10000:
            logger.warning(
                f"Shuffling RA with low ({num_of_events} events) statistics. Proceed with caution."
            )


def crawl_delay():
    """
    Delay between sending HTML requests.
    """

    time.sleep(np.random.uniform(5, 10))


def find_files(directory, keyword):
    """
    Find files in a directory that contain
    a keyword.
    """

    found_files = []

    for root, dirs, files in os.walk(directory):
        if files:
            for f in files:
                if keyword in f:
                    found_files.append(os.path.join(root, f))

    return found_files


def find_folders(directory, keyword):
    """
    Find subfolders in a directory that
    contain a keyword.
    """

    found_folders = []

    for root, dirs, files in os.walk(directory):
        if dirs:
            for d in dirs:
                if keyword in d:
                    found_folders.append(os.path.join(root, d))

    return found_folders
