## QEDtool: A Python package for numerical quantum information in quantum electrodynamics

### Description

`qedtool` is a Python-based object-oriented tool that allows users to calculate quantum information quantities from relativistic perturbative quantum electrodynamics (QED) at tree-level. It contains functions that define 3-vectors, 4-vectors, Dirac spinors, propagators and quantum states, that can be Lorentz transformed in their corresponding representations. With `qedtool`, users can calculate

* polarized tree-level QED Feynman amplitudes,
* scattering probabilities and quantum informational quantities for pure and mixed polarization states,
* the full emitted quantum state (with built-in functions for 2-to-2 particle scattering).

From the emitted quantum state, users can compute
* the differential cross section (2-to-2 scattering)
* the degree of two-particle entanglement of emitted states,
* $n$-particle Stokes parameters,
* single- and two-particle degree of polarization.

These quantities can be studied from arbitrary reference frames by means of Lorentz transformations. The documentation of `qedtool` is contained within our paper: 

[![DOI](http://img.shields.io/badge/arXiv%20preprint%20-DOI-lightblue.svg)](https://arxiv.org/)

The `notebooks/` directory contains Jupyter notebooks with examples as presented in our paper; from defining and performing operations vectors, spinors, and particles, to complete scattering processes. If you find `qedtool` useful in your research, please cite our paper.

### Installation

In addition to cloning `qedtool` from this GitHub repository, it can be installed using the following command line:
```
  pip install qedtool
```

### License

`qedtool` is licensed under the MIT license.

