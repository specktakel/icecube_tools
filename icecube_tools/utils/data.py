from re import A
import numpy as np
import os
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
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

icecube_data_base_url = "https://icecube.wisc.edu/data-releases"
data_directory = os.path.abspath(os.path.join(os.path.expanduser("~"), ".icecube_data"))

available_periods = ["IC40", "IC59", "IC79", "IC86_I", "IC86_II"]


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
        #TODO: protect from overwriting

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



class Uptime():
    """
    Class to handle calculations of detector live time.
    """

    def __init__(self):
        self.data = {}
        #Store start and end times of each period separately
        """
        self.times = ddict()
        for c, p in enumerate(available_periods):
            self.data[p] = np.loadtxt(os.path.join(
                data_directory,
                "20210126_PS-IC40-IC86_VII", 
                "icecube_10year_ps",
                "uptime",
                f"{p}_exp.csv")
            )
            self.times.add(self.data[p][0, 0], p, "start")
            self.times.add(self.data[p][-1, -1], p, "end")
        """
        self.times= np.zeros((len(available_periods), 2))
        for c, p in enumerate(available_periods):
            self.data[p] = np.loadtxt(os.path.join(
                data_directory,
                "20210126_PS-IC40-IC86_VII", 
                "icecube_10year_ps",
                "uptime",
                f"{p}_exp.csv")
            )
            self.times[c, 0] = self.data[p][0, 0]
            self.times[c, 1] = self.data[p][-1, -1]

            

    def time_span(self, period):
        """
        :param period: String of data period.
        :return: total time between start and end of data period.
        """

        time = self.data[period][-1, -1] - self.data[period][0, 0]
        time = time * u.d
        return time.to("year")


    def time_obs(self, period):
        """
        :param period: String of data period.
        :return: Return total observation time of data period.
        """

        intervals = self.data[period][:, 1] - self.data[period][:, 0]
        time = np.sum(intervals) * u.d
        time = time.to("year")
        return time


    def find_obs_time(self, **kwargs):
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

        if start < self.times[0, 0]:
            logger.warning("Start time outside of running experiment, setting to earliest possible time.")
            start = self.times[0, 0]

        p_start = np.searchsorted(self.times[:, 0], start)
        

        if end > self.times[-1, -1]:
            logger.info("End time outside of provided data set, sending an owl to Professor Trelawney")
            # Set to highest allowed value
            p_end = len(available_periods) - 1
            future = True
            
        else:    
            p_end = np.searchsorted(self.times[:, 1], end)
            future = False

        

        # repeat searchsorted procedure for the periods containing start/end:
        # add up all the detector uptime in those to get the resulting obs time
        # or... just go for 'reasonable approximation':
        # weigh the uptime in one period with the amount of time covered in that period
        # assumes downtime is distributed uniformly
        # since time_obs/time_span \approx 1, doesn't really matter anyway

        obs_times = {}
        if p_start == p_end and not future:
            fraction = duration / self.time_span(available_periods[p_start])
            t_obs = fraction * self.time_obs(available_periods[p_start])
            obs_times[available_periods[p_start]] = t_obs.value
        else:
            # find duration in start period:
            duration = ((self.times[p_start, 1] - start) * u.day).to("year")
            fraction = duration / self.time_span(available_periods[p_start])
            t_obs_start = fraction * self.time_obs(available_periods[p_start])
            obs_times[available_periods[p_start]] = t_obs_start.value
                        
            # now for the middle periods:
            for c_p in range(p_start+1, p_end):
                obs_times[available_periods[c_p]] = self.time_obs(available_periods[c_p]).value
            
            # end
            duration = ((end - self.times[p_end, 0]) * u.day).to("year")
            fraction = duration / self.time_span(available_periods[p_end])
            t_obs_end = fraction * self.time_obs(available_periods[p_end])
            obs_times[available_periods[p_end]] = t_obs_end.value

        return obs_times




        """
        #this ain't working
        for p_start, time in zip(available_periods, self.times):
            if start >= time[0]:
                break
        for p_end, time in reversed(zip(available_periods, self.times)):
            if end <= 
        """


class SimEvents():
    def __init__(self, path):
        #read in data, store in attributes
        self.path = path
        with h5py.File(path, "r") as f:
            self.true_energy = f["true_energy"][()]
            self.reco_energy = f["reco_energy"][()]
            self.ra = f["ra"][()]
            self.dec = f["dec"][()]
            self.ang_err = f["ang_err"][()]
            self.source_label = f["source_label"][()]

    


class RealEvents():
    def __init(self, path):
        self.path = path
        with h5py.File(path, "r") as f:
            self.reco_energy = f["reco_energy"][()]
            self.ra = f["ra"][()]
            self.dec = f["dec"][()]
            self.ang_err = f["ang_err"][()]




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
