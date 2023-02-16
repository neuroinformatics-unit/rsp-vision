[![Python Version](https://img.shields.io/pypi/pyversions/cellfinder.svg)](https://pypi.org/project/cellfinder)
[![Wheel](https://img.shields.io/pypi/wheel/cellfinder.svg)](https://pypi.org/project/cellfinder)
[![Development Status](https://img.shields.io/pypi/status/cellfinder.svg)](https://github.com/brainglobe/cellfinder)
[![Tests](https://img.shields.io/github/workflow/status/brainglobe/cellfinder/tests)](
    https://github.com/brainglobe/cellfinder/actions)
[![codecov](https://codecov.io/gh/brainglobe/cellfinder/branch/master/graph/badge.svg?token=s3MweEFPhl)](https://codecov.io/gh/brainglobe/cellfinder)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

# RSP vision
TODO: Add a description of the project

## Demo Usage
To test the functionalities that have been implemented so far, request the pre-processed data and store it locally in a folder called `allen_dff`.

Second, set up the environmental variables and the config file by executing:
```bash
python3 setup_for_demo.py
```
Then edit the config file with the correct paths to the data by overwriting `/path/to/`. The only path that matters at this stage is the `allen_dff` path, which should point to the folder where you stored the pre-processed data.

Finally, run the following commands with IPython:
```python
from load_suite2p.main import main
main()
```

This script will call the `main()` method and ask you for the name of the folder containing the data you want to analyse, which corresponds to a portion of the name of the data file.
