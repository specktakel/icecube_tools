from __future__ import annotations

from ..point_source_likelihood.point_source_likelihood import (
    TimeDependentPointSourceLikelihood,
)

from ..simulator import BackgroundSimulator
from ..utils.data import ddict, Events, Uptime, RealEvents, Uptime
from ..utils.coordinate_transforms import *

import yaml
import h5py
import healpy as hp
import numpy as np
from tqdm import tqdm as progress_bar

from abc import ABC, abstractmethod

import os.path
import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


"""
Classes to perform reproducible point source analyses
"""


class PointSourceAnalysis(ABC):
    """Meta class"""

    def __init__(self):
        pass

    @abstractmethod
    def load_config(self):
        pass

    @abstractmethod
    def write_config(self):
        pass

    @abstractmethod
    def write_output(self):
        pass

    @abstractmethod
    def apply_cuts(self):
        """Make cuts on loaded data"""
        pass

    @property
    def which(self):
        return self._which

    @classmethod
    def peek(cls, path: str):
        """
        Load previously saved hdf5 file
        :param path: Path to hdf5 file
        :param events: If provided, create `MapScan` and return it, else return `dict` with data
        """
        with h5py.File(path, "r") as f:
            dec_test = f["meta/dec"][()][0]
            return dec_test


class MapScan(PointSourceAnalysis):
    """
    Class for performing point source scans of the entire sky
    or some smaller patches.
    """

    # Config structure for yaml files
    config_structure = {
        "sources": {"nside": int, "npix": int, "ra": list, "dec": list},
        "data": {
            "periods": list,
            "cuts": {
                "northern": {"emin": float},
                "equator": {"emin": float},
                "southern": {"emin": float},
                "min_dec": float,
                "max_dec": float,
            },
            "likelihood": str,
        },
        "ts": {"seed": int, "ntrials": int},
    }

    def __init__(self, path: str, output_path: str, events: Events = None):
        """
        Instantiate analysis object. Parameters of the search have to be specified in a .yaml config file.
        Afterwards, source lists etc. can still be changed. A list of periods is not necessary;
        it is inferred from the provided events.
        :param path: Path to config
        :param events: object inheriting from :class:`icecube_tools.utils.data.Events`
        """

        self.load_config(path)
        if events is None:
            self.events = RealEvents.from_event_files(*self._data_periods)
            self._uptime = Uptime(*self._data_periods)
            self._irf_periods = self._uptime._irf_periods
        else:
            self.events = events
            self._irf_periods = self.events._irf_periods
            self._data_periods = self.events._data_periods
        self.uptime = Uptime(*self._data_periods)
        self.times = self.uptime.cumulative_time_obs()
        self.apply_cuts()

        self.output_path = output_path

    def perform_scan(self, show_progress: bool = False, minos: bool = False):
        """
        Perform scan over provided source list whose coordinates are stored in `self.ra_test, self.dec_test`
        :param show_progress: True if progress bar should be displayd
        :param minos: True if additionally `Minuit.minos()` should be called for calculating errors
        """
        # s = sched.scheduler(time.time, time.sleep)
        logger.info("Performing scan for periods: {}".format(self.events.periods))
        ra = self.events.ra
        dec = self.events.dec
        ang_err = self.events.ang_err
        reco_energy = self.events.reco_energy
        if show_progress:
            for c in progress_bar(range(len(self.ra_test))):
                self._test_source(
                    (self.ra_test[c], self.dec_test[c]),
                    c,
                    ra,
                    dec,
                    reco_energy,
                    ang_err,
                    minos,
                )
                if c % 60 == 59:
                    # refresh output file
                    self.write_output(self.output_path, source_list=True)
        else:
            for c, (ra_t, dec_t) in enumerate(zip(self.ra_test, self.dec_test)):
                self._test_source(
                    (ra_t, dec_t), c, ra, dec, reco_energy, ang_err, minos
                )
                if c % 60 == 59:
                    # refresh output file
                    self.write_output(self.output_path, source_list=True)
        self.write_output(self.output_path, source_list=True)

    def _test_source(
        self,
        source_coord: Tuple[float, float],
        num: int,
        ra: Dict,
        dec: Dict,
        reco_energy: Dict,
        ang_err: Dict,
        minos: bool = False,
    ):
        """
        Test source and store results in appropriate array.
        :param source_coord: Tuple of (ra, dec) in radians of test source
        :param num: Index of source, result of source is stored in self.some_array[num]
        :param ra: Dict with period as key, providing event RAs in radians
        :param dict: Dict with period as key, providing event DECs in radians
        :param reco_energy: Dict with period as key, providing reconstructed energy in GeV
        :param ang_err: Dict with period as key, providing 68% angular errors in degrees
        :param minos: True if `Minuit.minos()` should be called for calculating errors
        """

        try:
            self.likelihood.source_coord = source_coord
            if isinstance(self, MapScanTSDistribution):
                # Insert scrambled events if `self` is supposed to
                self.likelihood.reset_events(ra, dec, reco_energy, ang_err)
        except AttributeError as e:
            self.likelihood = TimeDependentPointSourceLikelihood(
                source_coord,
                self._data_periods,
                ra,
                dec,
                reco_energy,
                ang_err,
                which=self.which,
                times=self.times,
            )
        finally:
            if self.likelihood.Nprime > 0:  # else somewhere division by zero
                logging.debug("Nearby events: {}".format(self.likelihood.Nprime))
                self.ts[num] = self.likelihood.get_test_statistic()
                self.index[num] = self.likelihood._best_fit_index
                self.ns[num] = self.likelihood._best_fit_ns
                self.index_error[num] = self.likelihood.m.errors["index"]
                self.ns_error[num] = self.likelihood.m.errors["ns"]
                self.fit_ok[num] = self.likelihood.m.fmin.is_valid

                # is computationally too expensive for the entire grid, only use at certain points!
                if self.fit_ok[num] and minos:
                    minos = self.likelihood.m.minos()
                    if minos.valid:
                        self.index_merror[num, 0] = minos.merrors["index"].lower
                        self.index_merror[num, 1] = minos.merrors["index"].upper
                        self.ns_merror[num, 0] = minos.merrors["ns"].lower
                        self.ns_merror[num, 1] = minos.merrors["ns"].upper

            else:
                self.fit_ok[num] = True

    def load_config(self, path: str):
        """
        Load analysis config from file
        :param path: Path to config file
        """

        with open(path, "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)
        logger.debug("{}".format(str(config)))  # ?!
        self.config = config
        source_config = config.get("sources", False)
        if source_config:
            self.nside = source_config.get("nside")
            self.npix = source_config.get("npix")
            self.ra_test = source_config.get("ra")
            self.dec_test = source_config.get("dec")
            if self.ra_test and self.dec_test:
                if not isinstance(self.dec_test, list):
                    self.dec_test = [self.dec_test]
                if not isinstance(self.ra_test, list):
                    self.ra_test = [self.ra_test]
                self.ra_test = np.array(self.ra_test)
                self.dec_test = np.array(self.dec_test)
                assert self.ra_test.shape == self.dec_test.shape

        ts_config = config.get("ts")
        if ts_config:
            self.ntrials = ts_config["ntrials"]
            self.seed = ts_config["seed"]

        data_config = config.get("data")
        self._data_periods = data_config.get("periods")
        cuts = data_config.get("cuts", False)
        if cuts:
            self.northern_emin = float(
                data_config.get("cuts").get("northern").get("emin", 1e1)
            )
            self.equator_emin = float(
                data_config.get("cuts").get("equator").get("emin", 1e1)
            )
            self.southern_emin = float(
                data_config.get("cuts").get("southern").get("emin", 1e1)
            )
            self.min_dec = data_config.get("cuts").get("min_dec", -90)
            self.max_dec = data_config.get("cuts").get("max_dec", 90)
        else:
            self.northern_emin = 1e1
            self.equator_emin = 1e1
            self.southern_emin = 1e1
            self.min_dec = -90.0
            self.max_dec = 90.0
        self._which = data_config.get("likelihood", "both")

    def write_config(self, path: str, source_list: bool = False):
        """
        Write config used in analysis to file
        :param path: Path to config file
        :param source_list: True if source list (ra, dec) should be written to config
        """

        config = ddict()
        try:
            config.add(self.nside, "sources", "nside")
        except AttributeError:
            pass
        try:
            config.add(self.npix, "sources", "npix")
        except AttributeError:
            pass
        if source_list:
            config.add(self.ra_test.tolist(), "sources", "ra")
            config.add(self.dec_test.tolist(), "sources", "dec")
        config.add(self._data_periods, "data", "periods")
        try:
            for emin, region in zip(
                [self.northern_emin, self.equator_emin, self.southern_emin],
                ["northern", "equator", "southern"],
            ):
                config.add(emin, "data", "cuts", region, "emin")
        except AttributeError:
            pass
        try:
            config.add(self.min_dec, "data", "cuts", "min_dec")
        except AttributeError:
            config.add(-90, "data", "cuts", "min_dec")
        try:
            config.add(self.max_dec, "data", "cuts", "max_dec")
        except AttributeError:
            config.add(90, "data", "cuts", "min_dec")
        config.add(self.which, "data", "likelihood")

        try:
            config.add(self.ntrials, "ts", "ntrials")
        except AttributeError:
            pass
        try:
            config.add(self.seed, "ts", "seed")
        except AttributeError:
            pass

        with open(path, "w") as f:
            yaml.dump(config, f)

    def write_output(self, path: str, source_list: bool = False):
        """
        Save analysis results to hdf5 and additionally the used config, path is saved in results hdf5.
        :param path: Path to file
        :param source_list: True if source list should be written to additionally saved yaml config
        """

        try:
            self.ts
            # assert np.any(self.ts)
        except (AttributeError, AssertionError):
            logging.error("Call perform_scan() first")
            return

        self.write_config(os.path.splitext(path)[0] + ".yaml", source_list=source_list)

        with h5py.File(path, "w") as f:
            meta = f.create_group("meta")
            meta.create_dataset("ra", shape=self.ra_test.shape, data=self.ra_test)
            meta.create_dataset("dec", shape=self.dec_test.shape, data=self.dec_test)
            meta.create_dataset("periods", data=self._data_periods)
            meta.attrs["config_path"] = os.path.splitext(path)[0] + ".yaml"

            data = f.create_group("output")
            data.create_dataset("ts", shape=self.ts.shape, data=self.ts)
            data.create_dataset("index", shape=self.index.shape, data=self.index)
            data.create_dataset("ns", shape=self.ns.shape, data=self.ns)
            data.create_dataset(
                "ns_error", shape=self.ns_error.shape, data=self.ns_error
            )
            data.create_dataset(
                "index_error", shape=self.index_error.shape, data=self.index_error
            )
            data.create_dataset(
                "ns_merror", shape=self.ns_merror.shape, data=self.ns_merror
            )
            data.create_dataset(
                "index_merror", shape=self.index_merror.shape, data=self.index_merror
            )
            data.create_dataset("fit_ok", shape=self.fit_ok.shape, data=self.fit_ok)

    @classmethod
    def load_output(cls, path: str, events: Events = None):
        """
        Load previously saved hdf5 file
        :param path: Path to hdf5 file
        :param events: If provided, create `MapScan` and return it, else return `dict` with data
        """
        with h5py.File(path, "r") as f:
            if events is not None:
                config_path = f["meta"].attrs["config_path"]
                if not os.path.isfile(config_path):
                    config_path = os.path.splitext(path)[0] + ".yaml"
                obj = cls(config_path, path, events)
                obj.ts = f["output/ts"][()]
                obj.index = f["output/index"][()]
                obj.ns = f["output/ns"][()]
                obj.index_error = f["output/index_error"][()]
                obj.ns_error = f["output/ns_error"][()]
                obj.ns_merror = f["output/ns_merror"][()]
                obj.index_merror = f["output/index_merror"][()]
                obj.fit_ok = f["output/fit_ok"][()]
                obj.ra_test = f["meta/ra"][()]
                obj.dec_test = f["meta/dec"][()]
                return obj

            else:
                output = {}
                output["ts"] = f["output/ts"][()]
                output["index"] = f["output/index"][()]
                output["ns"] = f["output/ns"][()]
                output["index_error"] = f["output/index_error"][()]
                output["ns_error"] = f["output/ns_error"][()]
                output["ns_merror"] = f["output/ns_merror"][()]
                output["index_merror"] = f["output/index_merror"][()]
                output["fit_ok"] = f["output/fit_ok"][()]
                output["ra_test"] = f["meta/ra"][()]
                output["dec_test"] = f["meta/dec"][()]
                output["config_path"] = f["meta"].attrs["config_path"]
                return output

    def generate_sources(self, nside: bool = True):
        """
        Generate sources from config-specified specifics.
        Provided ra, dec lists take priority over npix and nside.
        :param nside: If healpy's nside should be used in calculating test sources.
        """

        reload = True
        if self.ra_test is not None and self.dec_test is not None:
            assert len(self.ra_test) == len(self.dec_test)
            logger.info("Using provided ra and dec")
            reload = False
        elif self.nside and nside:
            self.npix = hp.nside2npix(self.nside)
            logger.warning("Overwriting npix with nside = {}".format(self.nside))
        elif self.npix and not nside:
            logger.info("Using npix = {}".format(self.npix))

        if reload:
            logger.info(
                f"resolution in degrees: {hp.nside2resol(self.nside, arcmin=True)/60}"
            )
            theta_test, phi_test = hp.pix2ang(
                self.nside, np.arange(self.npix), nest=False
            )
            ra_test, dec_test = spherical_to_icrs(theta_test, phi_test)
            self.ra_test = ra_test[
                np.nonzero(
                    (dec_test <= np.deg2rad(self.max_dec))
                    & (dec_test >= np.deg2rad(self.min_dec))
                )
            ]
            self.dec_test = dec_test[
                np.nonzero(
                    (dec_test <= np.deg2rad(self.max_dec))
                    & (dec_test >= np.deg2rad(self.min_dec))
                )
            ]
        self._make_output_arrays()

    def _make_output_arrays(self):
        """
        Creates output arrays based on ra, dec lists.
        """

        if self.ra_test is not None and self.dec_test is not None:
            num = len(self.ra_test)
        else:
            raise ValueError(
                "Can't create output arrays, no well-defined source list supplied."
            )

        self.ts = np.zeros(num)
        self.index = np.zeros(num)
        self.ns = np.zeros(num)
        self.ns_error = np.zeros(num)
        self.index_error = np.zeros(num)
        self.fit_ok = np.zeros(num, dtype=bool)
        self.index_merror = np.zeros((num, 2))  # for asymmetric minos errors
        self.ns_merror = np.zeros((num, 2))  # for asymmetric minos errors

    def apply_cuts(self):
        """
        Apply cuts of energy and dec that are provided in the yaml config.
        Actual cuts are only applied to displayed data of `self.events`
        in terms of masked properties. All data stored in private variables
        stays in place.
        """

        mask = {}
        self.events.mask = None
        try:
            for p in self._irf_periods:
                events = self.events.period(p)
                mask[p] = np.nonzero(
                    (
                        (events["reco_energy"] > self.northern_emin)
                        & (events["dec"] >= np.deg2rad(10))
                    )
                    | (
                        (events["reco_energy"] > self.equator_emin)
                        & (events["dec"] < np.deg2rad(10))
                        & (events["dec"] > np.deg2rad(-10))
                    )
                    | (
                        (events["reco_energy"] > self.southern_emin)
                        & (events["dec"] <= np.deg2rad(-10))
                    )
                )
            self.events.mask = mask
        except AttributeError:
            pass

    '''
    @classmethod
    def combine_outputs(cls, config: str, events: Events, output_path: str, *paths) -> MapScan:
        """
        Combine multiple files to a single MapScan instance.
        Task of making sure that the order of files and coordinates sticks to the order of healpy is delegated to the user.
        :config: str, path to an arbitrary config file of the split analysis.
        :paths: Paths to outputs of analyses.
        :return: Instance of `MapScan`
        """

        logger.warning("Make sure that the files provided as arguments have the correct order!")

        for path in paths:
            cls.load_output(file)
        return cls(config, events, output_path)
    '''

    def make_p_values(self, file_base: str) -> np.ndarray:
        """
        Method to load output of `MapScanTSDistribution` and use it to create p_values from all TS values.
        :param file_base: str of file names common to all results of `MapScanTSDistribution` outputs
        :return: np.ndarray of p_values of shape `self.ts.shape`
        """

        from ..detector.effective_area import EffectiveArea

        p_values = np.zeros_like(self.ts)
        alpha = np.zeros_like(self.ts)

        aeff = EffectiveArea.from_dataset("20210126", "IC86_II")
        dec_bins = np.sort(np.arcsin(-aeff.cos_zenith_bins))
        decs = {}

        directory = os.path.dirname(file_base)
        files = os.listdir(directory)

        # Sort files by declination band of test source
        for file in files:
            if not ".hdf5" in file:
                continue
            dec = np.digitize(self.peek(os.path.join(directory, file)), dec_bins) - 1
            try:
                decs[dec].append(os.path.join(directory, file))
            except KeyError:
                decs[dec] = [os.path.join(directory, file)]
        # Go through all files and use the combined results to convert TS into p_value
        for dec, arr in decs.items():
            # Load outputs and combine
            output = MapScanTSDistribution.combine_outputs(*arr)
            # Find entries where the declination of test source is in the same bin as the declination of the simulated source
            idx = np.digitize(self.dec_test, dec_bins) - 1 == dec
            # print(idx)
            # Find position of selected TS in the simulations, divide by number of trials
            alpha[idx] = (
                np.digitize(self.ts[idx], np.sort(output["ts"])) - 1
            ) / output["ntrials"]
            # If the maximum TS from simulations does not exceed the data TS, use the largest possible alpha
            alpha[idx][alpha[idx] == 1.0] = (output["ntrials"] - 1) / output["ntrials"]
        p_values = 1.0 - alpha

        return p_values, alpha

    @classmethod
    def combine_outputs(cls, *paths, events: Events = None):
        """
        Wrapper for `load_output` to load and combine multiple data sets.
        The task of checking for the correct declination is delegated to the user. #TODO check if check  works
        :param paths: Paths to files whose output should be combined
        :return: Dict of combined data
        """

        # Should create data structure for all different declination bins that are found first
        ts = []
        index = []
        index_error = []
        index_merror = []
        ns = []
        ns_error = []
        ns_merror = []
        fit_ok = []
        ra_test = []
        dec_test = []
        ntrials = 0
        for path in paths:
            input = cls.load_output(path)
            ts.append(input["ts"])
            index.append(input["index"])
            index_error.append(input["index_error"])
            index_merror.append(input["index_merror"])
            ns.append(input["ns"])
            ns_error.append(input["ns_error"])
            ns_merror.append(input["ns_merror"])
            fit_ok.append(input["fit_ok"])
            ntrials += int(np.sum(input["fit_ok"]))
            if cls == MapScanTSDistribution:
                try:
                    assert np.isclose(declination, input["dec_test"])
                except NameError:
                    declination = input["dec_test"][0]
                    ra = input["ra_test"][0]
            else:
                ra_test.append(input["ra_test"])
                dec_test.append(input["dec_test"])

        if events is None:
            output = {}
            output["ts"] = np.hstack(ts)
            output["ns"] = np.hstack(ns)
            output["ns_error"] = np.hstack(ns_error)
            output["index"] = np.hstack(index)
            output["index_error"] = np.hstack(index_error)
            output["fit_ok"] = np.hstack(fit_ok)
            # Needs vstack because of different shape
            output["ns_merror"] = np.vstack(ns_merror)
            output["index_merror"] = np.vstack(index_merror)
            output["ntrials"] = ntrials
            if cls == MapScanTSDistribution:
                output["ra_test"] = np.array([ra])
                output["dec_test"] = np.array([declination])
            else:
                output["ra_test"] = np.hstack(ra_test)
                output["dec_test"] = np.hstack(dec_test)
            return output

        else:
            config_path = input["config_path"]
            output_path = os.path.join(os.path.dirname(config_path), "_output.hdf5")
            instance = cls(config_path, output_path, events)
            instance.ra_test = np.hstack(ra_test)
            instance.dec_test = np.hstack(dec_test)
            instance.ts = np.hstack(ts)
            instance.ns = np.hstack(ns)
            instance.ns_error = np.hstack(ns_error)
            instance.ns_merror = np.vstack(ns_merror)
            instance.index = np.hstack(index)
            instance.index_error = np.hstack(index_error)
            instance.index_merror = np.vstack(index_merror)
            instance.fit_ok = np.hstack(fit_ok)
            return instance


class MapScanTSDistribution(MapScan):
    """
    Class to create TS distributions (and subsequently local p-values.
    Inherhits from the 'normal' MapScan.
    """

    def __init__(
        self, path: str, output_path: str, events: Events = None, bg_sim: bool = False
    ):
        """
        Instantiate object to create a TS distribution at a given declination
        """

        super().__init__(path, output_path, events)
        self._make_output_arrays()
        # needs to be declination-dependent number of expected events
        # should have in each Aeff dec bin as much events as the real data
        # else we have some sort of bias against the effective area/data selection process
        # using the data driven background likelihood also circumvents the problem
        # of having to manually cut some data that's below the reco energy cut
        if bg_sim:
            self.sim = BackgroundSimulator(self._irf_periods[0])
            self.Nex = self.events.N[self._irf_periods[0]]

    def perform_scan(self, show_progress: bool = False, minos: bool = False):
        """
        Perform multiple (`ntrials`) fits of the same declination.
        :param show_progress: Bool, if True progress bar is shown, defaults to False
        :param minos: Bool, if True minos error calculation is carried out, defaults to False
        """

        logger.info("Performing scan for periods: {}".format(self.events.periods))
        self.events.seed = self.seed
        dec = self.events.dec
        reco_energy = self.events.reco_energy
        ang_err = self.events.ang_err
        if show_progress:
            for c in progress_bar(range(self.ntrials)):
                while True:
                    # repeat until a fit has converged
                    self.events.scramble_ra()
                    ra = self.events.ra
                    self._test_source(
                        (self.ra_test[0], self.dec_test[0]),
                        c,
                        ra,
                        dec,
                        reco_energy,
                        ang_err,
                        minos,
                    )
                    if self.fit_ok[c]:
                        # if converged, move to next iteration
                        break
                if c % 60 == 59:
                    # refresh output file
                    self.write_output(self.output_path, source_list=True)
        else:
            for c in range(self.ntrials):
                while True:
                    # repeat until a fit has converged
                    self.events.scramble_ra()
                    ra = self.events.ra
                    self._test_source(
                        (self.ra_test[0], self.dec_test[0]),
                        c,
                        ra,
                        dec,
                        reco_energy,
                        ang_err,
                        minos,
                    )
                    if self.fit_ok[c]:
                        # if converged, move to next iteration
                        break
                if c % 60 == 59:
                    # refresh output file
                    self.write_output(self.output_path, source_list=True)
        self.write_output(self.output_path, source_list=True)

    def perform_scan_bg_sim(self, show_progress: bool = False, minos: bool = False):
        """
        Perform multiple (`ntrials`) fits of the same declination.
        :param show_progress: Bool, if True progress bar is shown, defaults to False
        :param minos: Bool, if True minos error calculation is carried out, defaults to False
        """

        logger.info("Performing scan for periods: {}".format(self.events.periods))

        if show_progress:
            for c in progress_bar(range(self.ntrials)):
                while True:
                    self.sim.run(self.N, seed=self.seed + c)
                    self._test_source(
                        (self.ra_test[0], self.dec_test[0]),
                        self.sim.ra,
                        self.sim.dec,
                        self.sim.reco_energy,
                        self.sim.ang_err,
                        minos,
                    )
                    if self.fit_ok[c]:
                        # if converged, move to next iteration
                        break
                if c % 60 == 59:
                    # refresh output file
                    self.write_output(self.output_path, source_list=True)
        else:
            for c in range(self.ntrials):
                while True:
                    # repeat until a fit has converged
                    self.sim.run(self.N, seed=self.seed + c)
                    self._test_source(
                        (self.ra_test[0], self.dec_test[0]),
                        self.sim.ra,
                        self.sim.dec,
                        self.sim.reco_energy,
                        self.sim.ang_err,
                        minos,
                    )
                    if self.fit_ok[c]:
                        # if converged, move to next iteration
                        break
                if c % 60 == 59:
                    # refresh output file
                    self.write_output(self.output_path, source_list=True)
        self.write_output(self.output_path, source_list=True)

    def _make_output_arrays(self):
        """
        Creates output arrays based on ntrials
        """

        shape = self.ntrials
        self.ts = np.zeros(shape)
        self.index = np.zeros(shape)
        self.ns = np.zeros(shape)
        self.ns_error = np.zeros(shape)
        self.index_error = np.zeros(shape)
        self.fit_ok = np.zeros(shape)
        self.index_merror = np.zeros((shape, 2))
        self.ns_merror = np.zeros((shape, 2))
