# Overview

3D reconstruction from 2D projection images in Python using `PyTorch`.

*libtilt* provides Python implementations of

1. 2D projection of 3D volumes by
    - sampling along projection lines in real space 
   (*cf.* [Radon transform](https://en.wikipedia.org/wiki/Radon_transform))
    - sampling from the discrete Fourier transform (cubic volumes only)
   (*cf.* [central slice theorem](https://en.wikipedia.org/wiki/Projection-slice_theorem))

2. reconstructing 3D volumes from 2D projection images by
    - weighted backprojection
    - direct Fourier inversion (cubic volumes only)

These methods form the backbone of 3D reconstruction in electron cryomicroscopy, 
both single particle analysis and electron tomography. 
More information can be found 
in [this book chapter](https://link.springer.com/chapter/10.1007/978-1-4757-2163-8_5).


