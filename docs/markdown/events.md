# Event class

With the classes `SimEvents` and `RealEvents`, there are two containers for events, obviously distinguished by their respective origin - simulation or the actual detector.

Both classes include methods to load data from various (two) file types.


## Simulated events

From the other example notebook, we will simply load one of the simulated event files. 

```python
from icecube_tools.utils.data import RealEvents, SimEvents, data_directory, IceCubeData
from os.path import join
```

```python
path = "data/test_sim_IC86_I.h5"
events = SimEvents.read_from_h5(path)
events.ang_err[:5], events.source_label[:5]
```

The available data fields are:

```python
vars(events).keys()
```

Just as the simulation's output.


## Real events

For the real events, we load one of the provided data files of the 10 year point source release.
First, fetch the data set, then load one of the files.

```python
datainterface = IceCubeData()
data = datainterface.find("20210126")
datainterface.fetch(data)
```

```python
path = join(datainterface.data_directory, data[0].rstrip(".zip"), "icecube_10year_ps", "events", "IC86_II_exp.csv")
```

```python
events = RealEvents.read_from_file(path)
```

```python
vars(events).keys()
```

```python

```
