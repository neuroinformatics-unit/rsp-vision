# load-suite2p
This package focuses on loading 2p / 3p data generated by suite2p or registers 2p.
Part of a bigger project to reorganize, improve and expand the code contained in [ctsitou_rsp_vision](https://github.com/SainsburyWellcomeCentre/ctsitou_rsp_vision).

Could work as a package to be called in a python script or as a CLI application. Its main purpose is to crate an object to be read by the `analysis` package.
## Structure of the package folder `load_suite2p`:
```
  __init__.py
  config/
      config.yml
  folder_naming_specs.py
  formatted_data.py
  main.py
  parsers/
      __init__.py
      chryssanthi.py
      parser.py
  read_config.py
  utils.py
```
## Description:
* `main()` in `main.py` can be used as the entry point of a CLI application. It takes care of logging with `fancy_log`. It is now used only for testing purposes, it will be expanded in the future.
* `read_config.py` contains a method to read yaml configs, stored in the `config/` folder. Will be expanded as the config file becames more complex.
* `utils.py` contains helper methods that I could use in multiple locations. I might move them in a separate repository in the future. It contains an helper method for `fancy_log`, two methods to check the connection to winstor (`can_ping_swc_server`, `is_winstor_mounted`) and `exception_handler` an exception wrapper for main.py.
* `formatted_data.py` is a draft of the class describing the object to be saved.
* `folder_name_specs.py` contains a class that holds the folder name in which experimental data is saved, details on this experiment extracted from the folder and paths. Recives as input the name of the folder and checks if it is valid and if the data can be read. Calls a `parser` to extract the details from the folder name which are specific to the scientist/project.
* `parsers/` contains the `parser` class and a parser tailor made for Chryssanthi's folder structure. The `parser` class is an abstract class that defines the methods that a parser should have. The `chryssanthi` parser is the only one implemented so far. It extracts the details from the folder name taking into account various exceptions in the formatting. It is called by `folder_name_specs.py`.
