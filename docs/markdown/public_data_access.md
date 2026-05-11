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

# Data access

IceCube has a bunch of public datasets available at [https://icecube.wisc.edu/science/data-releases/](https://icecube.wisc.edu/science/data-releases/). `icecube_tools` provides an easy interface to this repository so that you can download and organise your data through python.

```python
from icecube_tools.utils.data import IceCubeData
```

The `IceCubeData` class provides this functionality.

```python
my_data = IceCubeData()
```

You can use the `find` method to pick out datasets you are interested in.

```python
found_dataset = my_data.find("20181018")
found_dataset
```

The `my_data` object has been inititalised to store data in the package's default location ("~/.icecube_data"). This is where other `icecube_tools` modules will look for stored data.

```python
my_data.data_directory
```

The `fetch` method will download this dataset to this default location for later use by `icecube_tools`. This method takes a list of names, so can also be used to download multiple datasets. `fetch` has a built in delay of a few seconds between HTTP requests to avoid spamming the website. `fetch` will not overwrite files by default, but this can be forced with `overwrite=True`.

```python
# my_data.fetch(found_dataset)
```

You may not want to use `icecube_tools` for other stuff, so you can also fetch to a specificed location with the keyword `write_to`.

```python
# my_data.fetch(found_dataset, write_to="data", overwrite=True)
```

For convenience, there is also the `fetch_all_to` method to download all the available data to a specified location. We comment this here as it can take a while to execute.

```python
# my_data.fetch_all_to("data")
```
