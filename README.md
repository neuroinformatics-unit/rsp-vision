[![Python Version](https://img.shields.io/pypi/pyversions/cellfinder.svg)](https://pypi.org/project/cellfinder)
[![Wheel](https://img.shields.io/pypi/wheel/cellfinder.svg)](https://pypi.org/project/cellfinder)
[![Development Status](https://img.shields.io/pypi/status/cellfinder.svg)](https://github.com/brainglobe/cellfinder)
[![Tests](https://img.shields.io/github/workflow/status/brainglobe/cellfinder/tests)](
    https://github.com/brainglobe/cellfinder/actions)
[![codecov](https://codecov.io/gh/brainglobe/cellfinder/branch/master/graph/badge.svg?token=s3MweEFPhl)](https://codecov.io/gh/brainglobe/cellfinder)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

# MultiPhoton RSP Analysis
TODO: Add a description of the project

## Demo Usage
To test the functionalities that have been implemented so far, you can use `python3` from the command line and run the following commands:
```python
from load_suite2p.main import main
main()
```
This script will call the `main()` method and ask you for the name of the folder containing the data you want to analyse. Be sure to have the data locally or to have access to the folder in the shared server. If you don't, I can share the data with you privately.

The precise path to the data should be stored in a `.env` file that you can create in bash using the command `touch .env`. You can then edit using the command `nano .env`. The `.env` file could contain the following line: `CONFIG_PATH=config/config.yml`, which specifies that the config file, containing the path information, is located in the `config` folder, a subdirectory of the `load_suite2p` folder.

Here an example of the content of the config file:

```yaml
parser: Parser2pRSP

paths:
  imaging: '/path/to/imaging'
  allen-dff: '/path/to/imaging/allen_dff'
  serial2p: '/path/to/imaging/serial2p'
  stimulus-ai-schedule: '/path/to/imaging/stimulus_AI_schedule_files'

use-allen-dff: true
analysis-type: 'sf_tf'

```
