# MLEF

This repository contains the code used in Enomoto and Nakashita (2023). 


## Source

Prerequisites: Numpy and Scipy.

### Benchmark functions

- booth.py: Booth function
- rosenbrock.py: Rosenbrock function

- newton.py: exact Newton optimization
- mlef.py:
- mlef_zeta.py: 

### Single wind speed assimilation

A single wind speed assimulation described in Bowler et al. (2013).

- wind.py: driver

### Cycled experiments with a Korteweg&ndash;de Vries&ndash;Burgers equation (KdVB) model

Assimilation into a KdVB equation model in Zupanski (2005).

#### Model scripts

- ode.py: 4th order Runge-Kutta
- kdvb.py: Korteweg&ndash;de Vries&ndash;Burgers equation model

#### Data assimilation scripts

Run gentrue.py, genobs.py and genens.py before cycle.py.
Edit cycle.py for parameters and a choice of an observation operater.

- hop.py: linear and nonlinear observation operators
- gentrue.py: generate the true run
- genobs.py: add observation to the true run
- genens.py: generate an ensemble
- cycle.py: driver

## References

- Bowler, N. E., J. Flowerdew, and S. R. Pring, 2013: Tests of different flavours of EnKF on a simple model. *Quart. J. Roy. Meteor. Soc.*, **139**, 1505&ndash;1519, [doi:10.1002/qj.2055](https://doi.org/10.1002/qj.2055).
- Enomoto, T., and S. Nakashita, 2024: Application of exact Newton optimisation to the maximum likelihood ensemble filter. *Tellus A*, **76**, 42–56, [doi:10.16993/tellusa.3255](https://doi.org/10.16993/tellusa.3255).
- Zupanski, M., 2005: Maximum Likelihood Ensemble Filter: Theoretical aspects. *Mon. Wea. Rev.*, **133**, 1710&ndash;1726, [doi:10.1175/MWR2946.1](https://doi.org/10.1175/MWR2946.1).
