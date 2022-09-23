from icecube_tools.utils.data import RealEvents, SimEvents, data_directory, IceCubeData
from os.path import join

my_data = IceCubeData()


def test_data_scan():

    assert my_data.datasets[1] == "20080911_AMANDA_7_Year_Data.zip"


def test_file_download(output_directory):

    found_dataset = my_data.find("AMANDA")

    my_data.fetch(found_dataset, write_to=output_directory)


def test_sim_event_container():
    path = "../docs/markdown/data/test_sim_IC86_I.h5"   # does this work?
    events = SimEvents.read_from_h5(path)
    events.ang_err[:5], events.source_label[:5]


def test_real_event_container():
    datainterface = IceCubeData()
    data = datainterface.find("20210126")
    datainterface.fetch(data)
    path = join(datainterface.data_directory, data[0].rstrip(".zip"), "icecube_10year_ps", "events", "IC86_II_exp.csv")
    events = RealEvents.read_from_file(path)