# Regridding for DAMASK


Demonstrate regridding capabilites of the DAMASK grid solver.

This repository contains the following scripts:

## [mappings.py](mappings.py)
Prints the relationships needed to map data to the new grid.

## [double\_resolution.py](double_resolution.py)
Maps existing results to a grid with doubled resolution and exports to VTK.

## [mesh\_replacement.py](mesh_replacement.py)
Implements the mesh replacement method for large deformation simulations.

## References
- K. Sedighiani, K. Traka, F. Roters, J. Sietsma, D. Raabe, and M. Diehl.
  Crystal plasticity simulation of in-grain microstructural evolution during large deformation of IF-steel.
  Acta Materialia 237:118167, 2022. [doi:10.1016/j.actamat.2022.118167](https://doi.org/10.1016/j.actamat.2022.118167)
