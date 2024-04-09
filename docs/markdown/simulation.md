---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: icecube_dev
    language: python
    name: python3
---

# Simulation

We can bring together the detector and source modelling to calculate the expected number of neutrino events and run simulations.

```python
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import h5py
```

## Defining a source and detector model

```python
from icecube_tools.detector.effective_area import EffectiveArea
from icecube_tools.detector.energy_resolution import EnergyResolution
from icecube_tools.detector.angular_resolution import AngularResolution
from icecube_tools.detector.detector import IceCube
from icecube_tools.source.flux_model import PowerLawFlux
from icecube_tools.source.source_model import DiffuseSource, PointSource
from icecube_tools.detector.r2021 import R2021IRF
from icecube_tools.utils.data import SimEvents
```

```python
# Define detector (see detector model notebook for more info)
aeff = EffectiveArea.from_dataset("20210126", "IC86_II")
irf = R2021IRF.from_period("IC86_II")
#IceCube expects an instance of EffectiveAerea, AngularResolution and
#EnergyResolution, optionally the period (here IC86_I)
#R2021IRF inherits from AngularResolution and EnergyResolution
#just to be able to be used as both
detector = IceCube(aeff, irf, irf, "IC86_II")
```

```python
# Define simple sources (see source model notebook for more info)
diff_flux_norm = 3e-21 # Flux normalisation in units of GeV^-1 cm^-2 s^-1 sr^-1
point_flux_norm = 5e-19 # Flux normalisation in units of GeV^-1 cm^-2 s^-1 
norm_energy = 1e5 # Energy of normalisation in units of GeV
min_energy = 1e2 # GeV
max_energy = 1e8 # GeV

diff_power_law = PowerLawFlux(diff_flux_norm, norm_energy, 3.7, 
                              min_energy, max_energy)
diff_source = DiffuseSource(diff_power_law, z=0.0)

point_power_law = PowerLawFlux(point_flux_norm, norm_energy, 2.5, 
                               min_energy, max_energy)
point_source = PointSource(point_power_law, z=0., coord=(np.pi, np.deg2rad(30)))
sources = [diff_source, point_source]
```

## Expected number of neutrino events

Sometimes we just want to predict the number of events from sources in a detector without specifying all detector properties or running a simulation. We can do this with the `NeutrinoCalculator`. For this, we just need a source list and an effective area.

```python
from icecube_tools.neutrino_calculator import NeutrinoCalculator, PhiSolver
```

```python
nu_calc = NeutrinoCalculator(sources, aeff)
nu_calc(time=1, # years
        min_energy=min_energy, max_energy=max_energy, # energy range
        min_cosz=-1, max_cosz=1) # cos(zenith) range
```

The calculator returns a list of expected neutrino event numbers, one for each source.

We may also want to do the inverse, and find the `PointSource` flux normalisation corresponding to an expected number of events. For this there is the `PhiSolver` class.

```python
phi_solver = PhiSolver(aeff, norm_energy, min_energy, max_energy, 
                       time=1, min_cosz=-1, max_cosz=1)
phi_norm = phi_solver(Nnu=15, 
                      dec=-30, # degrees
                      index=2.0) # spectral index
phi_norm # GeV^-1 cm^-2 s^-1
```

## Set up and run simulation

```python
from icecube_tools.simulator import Simulator, TimeDependentSimulator
```

```python
# Set up simulation
simulator = Simulator(sources, detector, "IC86_II")
simulator.time = 1 # year

# Run simulation
simulator.run(show_progress=True, seed=42)
```

This way, the simulator calculates the expected number of neutrinos from these sources given the observation period, effective area and relevant source properties. We note that we could also run a simulation for a fixed number of neutrinos if we want, simply by passing the optional argument `N` to `simulator.run()`.

```python
simulator.write_to_h5("h5_test.hdf5", sources)
```

```python
simulator.arrival_energy
```

```python
events = SimEvents.load_from_h5("h5_test.hdf5")
```

```python
events.period("IC86_II").keys()
```

```python
len(events)
```

```python
"""
for i in [1.5, 2.0, 2.5, 3.0, 3.5, 3.7]:
    norm_energy = 1e5 # Energy of normalisation in units of GeV
    min_energy = 1e2 # GeV
    max_energy = 1e8 # GeV
    phi_solver = PhiSolver(aeff, norm_energy, min_energy, max_energy, 
                           time=1, min_cosz=-1, max_cosz=1)
    phi_norm = phi_solver(Nnu=2000, 
                          dec=30, # degrees
                          index=i) # spectral index
    phi_norm # GeV^-1 cm^-2 s^-1
    point_power_law = PowerLawFlux(phi_norm, norm_energy, i, 
                                   min_energy, max_energy)
    point_source = PointSource(point_power_law, z=0., coord=(np.pi, np.deg2rad(30)))
    sources = [point_source]
    simulator = Simulator(sources, detector)
    simulator.time = 1 # year

    # Run simulation
    simulator.run(show_progress=True, seed=42)
    simulator.save(f"data/sim_output_{i:.1f}.h5")
"""
```

```python
# Plot energies
bins = np.geomspace(1e2, max_energy)
fig, ax = plt.subplots()
ax.hist(events.true_energy["IC86_II"], bins=bins, alpha=0.7, label="E_true")
ax.hist(events.reco_energy["IC86_II"], bins=bins, alpha=0.7, label="E_reco")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("E [GeV]")
ax.legend()
```

```python
# Plot directions
ps_sel = events.source_label["IC86_II"] == 1

fig, ax = plt.subplots(subplot_kw={"projection": "aitoff"})
fig.set_size_inches((12, 7))

circles = []
for r, d, a in zip(events.ra["IC86_II"][~ps_sel], events.dec["IC86_II"][~ps_sel], events.ang_err["IC86_II"][~ps_sel]):
    circle = Circle((r-np.pi, d), radius=np.deg2rad(a))
    circles.append(circle)
df_nu = PatchCollection(circles)

circles = []
for r, d, a in zip(events.ra["IC86_II"][ps_sel], events.dec["IC86_II"][ps_sel], events.ang_err["IC86_II"][ps_sel]):
    circle = Circle((r-np.pi, d), radius=np.deg2rad(a))
    circles.append(circle)
ps_nu = PatchCollection(circles, color="r")

ax.add_collection(df_nu)
ax.add_collection(ps_nu)

ax.grid()
fig.savefig("example_simulation.png", dpi=150)
```

# Time dependent simulation

We can simulate an observation campaign spanning multiple data periods of IceCube through a "meta class" `TimeDependentSimulator`:

```python
tsim = TimeDependentSimulator(["IC86_I", "IC86_II"], sources)
```

We need to set simulation times for all periods. Since for past periods the simulation time shouldn't be larger than the actual observation time (that is time span - down time of the detector) we need to take care, or rather, we let the class `Uptime` take care:

```python
from icecube_tools.utils.data import Uptime
```

It lets us calculate the actual observation time through, e.g. IC86_II, vs the time covered:

```python
uptime = Uptime("IC86_I", "IC86_II")
uptime._time_obs("IC86_II"), uptime.time_span()["IC86_II"]
```

We can further define a start and end time of an observation and let it calculate the observation time in each period. Viable possible options are
 - start and end time in MJD
 - start time in MJD and duration in years
 - end time in MJD and duration in years

If the start time is before the earliest period (IC_40), the start time will be set to the earliest possible date.

If the end time is past the last period (IC86_II), then we get an according message and simulate into the future.

We can of course bypass this time setting and operate directly on the instances of Simulator, for example if we'd want to build up large statistics for subsequent likelihood analysis.

```python
times = uptime.find_obs_time(start=55869, duration=3)
times
```

The returned dictionary can be used to set the simulation times for an instance of `TimeDependentSimulator`:

```python
tsim.time = times
```

The simulation is started by calling `run()`, results can be saved by `save(file_prefix)`, with the filename being `{file_prefix}_{p}.h5` with period `p`.

```python
tsim.run(show_progress=True)

```

```python
tsim._true_energy
```

```python
tsim.write_to_h5("multi_test.hdf5", sources)
```

```python
tsim.arrival_energy
```

```python
events = SimEvents.load_from_h5("multi_test.hdf5")
```

```python
events.arrival_energy

```

```python
"""
for i in [3.9]:
    norm_energy = 1e5 # Energy of normalisation in units of GeV
    min_energy = 1e2 # GeV
    max_energy = 1e8 # GeV
    phi_solver = PhiSolver(aeff, norm_energy, min_energy, max_energy, 
                           time=1, min_cosz=-1, max_cosz=1)
    phi_norm = phi_solver(Nnu=2000, 
                          dec=30, # degrees
                          index=i) # spectral index
    phi_norm # GeV^-1 cm^-2 s^-1
    point_power_law = PowerLawFlux(phi_norm, norm_energy, i, 
                                   min_energy, max_energy)
    point_source = PointSource(point_power_law, z=0., coord=(np.pi, np.deg2rad(30)))
    sources = [point_source]
    tsim = TimeDependentSimulator(["IC86_I", "IC86_II"], sources)
    for sim in tsim.simulators.values():
        sim.time = 1
    tsim.run(show_progress=True)
    tsim.save(f"index_{i}")
"""
```

```python

```
