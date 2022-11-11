# gpu_xray_scattering
Python package for calculating X-ray scattering on GPU

In this package we modify the XSNAMD package to implement python binding that allows for X-ray scattering calculations using GPUs.

## Install

`pip install -e .`

You need a working nvcc compiler.

## Testing

`python test/test.py`

## Actual usage

`python examples/1L2Y_example.py`

## Benchmarking

There is an overhead of 2.5 ms per calculation due to matrix copying from host memory to device memory. This becomes negligible when protein size is larger.


