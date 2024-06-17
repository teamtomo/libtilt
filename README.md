# libtilt

[![License](https://img.shields.io/pypi/l/libtilt.svg?color=green)](https://github.com/alisterburt/libtilt/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/libtilt.svg?color=green)](https://pypi.org/project/libtilt)
[![Python Version](https://img.shields.io/pypi/pyversions/libtilt.svg?color=green)](https://python.org)
[![CI](https://github.com/alisterburt/libtilt/actions/workflows/ci.yml/badge.svg)](https://github.com/alisterburt/libtilt/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/alisterburt/libtilt/branch/main/graph/badge.svg)](https://codecov.io/gh/alisterburt/libtilt)

Image processing for cryo-electron tomography in PyTorch


## For developers

We advise to fork the repository and make a local clone of your fork. After setting
up an environment (e.g. via miniconda), the development version can be installed with
(`-e` runs an editable install):

```commandline
python -m pip install -e '.[dev,test]'
```

Then ready pre-commits for automated code checks and styling:

```commandline
pre-commit install
```

Before making any pull request please make sure all the unittests pass:

```commandline
python -m pytest
```

The libtilt `device_test` decorator should be added to each test and tries to run
the code on GPU. This can either be via CUDA or via the 'mps' backend for M1 chips.
Please keep in mind that without a GPU available on your system, the code is not
explicitly tested to run on GPU.
