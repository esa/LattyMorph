# LattyMorph

Totimorphic structures are a type of neutrally-stable lattices that can change their geometry via simple mechanical actuations (e.g., rotation of beam and lever elements).
They are composed of unit cells that consist of a beam element, a lever connected to the middle of the beam, and zero-length springs connecting the two ends of the beam and the lever.
For more details, see the [original paper](https://doi.org/10.1073/pnas.2107003118) on Totimorphic structures.

This library provides the following features:

- 2D and 3D pyTorch models of Totimorphic structures which can be optimised using automatic differentiation. In particular, we introduce generalized coordinates to model the lattices.
- Finite element code for calculating mechanical properties of 2D Totimorphic structures. Includes an optimisation framework for continuously inversely designing a structure.
- Optics code for prototyping a mirror telescope made from a Totimorphic lattice, capable of continuously changing its shape to focus reflected light in a desired point (e.g., it can change its focal length).

Authors: Dominik Dold, Nicole Rosi
Special thanks to: Amy Thomas

## Getting started

Install the package in developer mode using

```
pip install -e .
```

in the repo folder (where setup.py is located).

Requires pyTorch >= '1.13.1+cu117'.

## Experiments

Some Jupyter notebooks illustrating how to use the package (e.g., in order to find the trajectory from configuration A to B of a lattice) are in the `Experiments` folder.

## Unittests

Some rudimentary unit tests can be run using

```
python unittests.py
```

in the repo folder.

## Citation

If you use the provided data set or code, or find it helpful for your own work, please cite

```
@article{dold2024continuous,
  title={Continuous Design and Reprogramming of Totimorphic Structures for Space Applications},
  author={Dominik Dold and Amy Thomas and Nicole Rosi and Jai Grover and Dario Izzo},
  year={2024},
  eprint={2411.15266},
  archivePrefix={arXiv},
  primaryClass={astro-ph.IM},
  url={https://arxiv.org/abs/2411.15266}, 
}
```

## Related work

Check out [pyLattice2D](https://gitlab.com/EuropeanSpaceAgency/pylattice2d) for pyTorch code enabling inverse design of irregular lattices using automatic differentiation.
