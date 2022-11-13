# gpu_xray_scattering
Python package for calculating X-ray scattering on GPU

In this package we modify the XSNAMD package to implement python binding that allows for X-ray scattering calculations using GPUs.

## Install

```bash
cd gpu_xray_scattering
make # Generates bin/XS.so
pip install -e .
```

You need a working nvcc compiler.

## Testing

`python test/test.py`

## Actual usage

Please consult `python examples/1L2Y_example.py`

## Parameters in scattering

The scattering calculator object

`scatter = Scatter(q=np.linspace(0, 1, 200), c1=1.0, c2=2.0, r_m=1.62, sol_s=1.8, num_raster=512, rho=0.334)`

has a number of parameters. They are explained below.

**`q`**

`q` is the scattering vector.

**Fitting parameters**

In scattering calculations of biomolecules in solution, there are two factors people change to fit the data.
One (`c1`) is called solvent exclusion term. 
This term removes a major part of the *in vacuo* form factor, modeling the remaining value as the excess form factor compared to the solvent background. 
It defaults to `1.0`.
The other term is called `c2`, which denotes solvation shell factor, which defaults to 2.0 (sensible values can be from -2 to 4).
In short, the surface of the biomolecule has partial charges that attract solvent molecules with dipoles.
This effectively increases the electron density on the surface of the biomolecule, making them scatter more X-ray and "look larger".
This factor correlates to how much (on average) solvent is attracted to the surface of the biomolecule.

**Other minor parameters**

`r_m` is the mean atomic radius used in calculation related to `c1`.
This value is set as the same as in Svergun's CRYSOL implementation (1.62 Angstroms).

`sol_s` is the solvent sphere used to determine surface-accessible surface area (SASA).
It can vary from 1.4 to 1.8 (Angstroms) but the result really won't change that much.

`num_raster` is the number of raster points also used for SASA calculation.
High `num_raster` numbers correlates to higher accuracy at the expense of speed.

`rho` is the electron density (0.334 electrons per Angstrom cubed at 20 degree C for pure water).
This value should be adjusted to the actual temperature at measurement.
The calculation of such value is left for users as a homework assignment.

## Benchmarking

There is an overhead of 2.5 ms per calculation due to matrix copying from host memory to device memory. This becomes negligible when protein size is larger.

![benchmark](benchmarking.png)

Benchmarking shows this code is strictly O(N^2) scaling, which is not good for very large N.
Considering that N could be larger than 1.5e5 atoms, I'm working on a O(QN) method based on orientational averaging.

