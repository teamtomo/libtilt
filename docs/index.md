# Overview

[![License](https://img.shields.io/pypi/l/libtilt.svg?color=green)](https://github.com/alisterburt/libtilt/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/libtilt.svg?color=green)](https://pypi.org/project/libtilt)
[![Python Version](https://img.shields.io/pypi/pyversions/libtilt.svg?color=green)](https://python.org)
[![CI](https://github.com/alisterburt/libtilt/actions/workflows/ci.yml/badge.svg)](https://github.com/alisterburt/libtilt/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/alisterburt/libtilt/branch/main/graph/badge.svg)](https://codecov.io/gh/alisterburt/libtilt)

*libtilt* provides Python implementations of methods for 

- **reconstructing 3D images from 2D projections** 
- **making 2D projections of 3D images**

This package is implemented using [*PyTorch*](https://pytorch.org/).

## methods

### reconstruction

- weighted backprojection
- direct Fourier inversion

### projection

- sampling along projection lines in real space 
(*cf.* [Radon transform](https://en.wikipedia.org/wiki/Radon_transform))
- sampling from the discrete Fourier transform
(*cf.* [central slice theorem](https://en.wikipedia.org/wiki/Projection-slice_theorem))

--- 

More information on the methods implemented here can be found 
in [this book chapter](https://link.springer.com/chapter/10.1007/978-1-4757-2163-8_5).
These methods form the basis of 
[3D reconstruction in electron cryomicroscopy](https://academic.oup.com/jmicro/article/65/1/57/2579723), 
both single particle analysis and electron tomography. 



