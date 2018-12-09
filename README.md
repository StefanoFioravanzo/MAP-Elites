## MAP-Elites for Constrained Optimization

Python implementation of the MAP Elites algorithm.

**TODO**: Add short explanation of MAP Elites

#### Implementation

```
.
├── controller.py
├── map_elites
│   ├── mapelites.py
│   ├── plot_utils.py
│   └── ea_operators.py
├── functions.py
├── config.ini
├── mapelites_continuous_opt.py
├── notebooks
│   ├── plotter.ipynb
│   └── n_dim_plotter.ipynb
└── utils
    └── fcnsuite.c
```

The algorithm is implemented to be as general as possible and applicable to any setting. The main MAPElites class in `mapelites.py` is an abstract class implementing all the base logic common to any instance of the algorithm.  
To run the algorithm you need to subclass `MapElites` and implement `map_x_to_b()`, `performance_measure()`, `generate_random_solution()`, `generate_feature_dimensions()`. Refer to the functions documentation and the example below for more details.

**TODO**: Talk about FeatureDimension class

MAP-Elites was designed to have a visual impact by showing its results in a heatmap plot in the feature dimensions. This implementation supports up to 4 dimensional heatmap plotting, with nested dimensions in the same plot. Refer to the `plot_heatmap()` function in `plot_utils.py` for more details.

Crossover and mutation evolutionary operators are implemented in `ea_operators.py`, you can extend that source file to add more custom evolutionary operators.

#### Configuration

All the configuration can be done using the `config.ini` file provided at the root of the project. Here is an example configuration file:

```ini
[mapelites]
; random seed
seed = 54
; number of initial random samples
bootstrap_individuals = 100
; numer of map elites iterations
iterations = 10000
; True: solve a minimization problem. False: solve a maximization problem
minimization = True

[opt_function]
; Define the optimization function.
; This must be the name of a class subclassing the abstract class ConstrainedFunction. See functions.py for reference
name = C16
; Number of dimensions of the optimization function
dimensions = 4
; Define heatmap bins for feature dimensions
; Name each bin as `bin_{name of constraint}` where `name_of_constraint` is the name of the constraint
; function implemented in the specified optimization function class
; If you want to define ONE bin for all constraints, name it `bin_all`
; Note: The bins must be defined by numbers, except for the `inf` label which can be defined ether at the beginning
; of at the end of the bins.
; bin_all = inf,0.0,1.0,2.0,3.0,4.0,inf
bin_g1 = inf,0.0,1.0,2.0,3.0,4.0,inf
bin_g2 = inf,0.0,1.0,2.0,3.0,inf
bin_h1 = inf,0.0,1.0,2.0,3.0,4.0,5.0,inf
bin_h2 = inf,0.0,1.0,2.0,inf

[crossover]
; Crossover function taken from ea_operators.py file.
; Name of called function is {type}_crossover(). If `type = UNIFORM` then the function call is `uniform_crossover()`
type = UNIFORM
; list of arguments to the above function
indpb = 0.5

[mutation]
; mutation function taken from ea_operators.py file.
; name of called function is {type}_mutation(). If `type = GAUSSIAN` then the function call is `gaussian_mutation()`
type = GAUSSIAN
; list of arguments to the above function
mu = 0
sigma = 0.2
; probability of each attribute to be mutated
indpb = 0.5
```

## Example: Continuous Constrained Optimization

In this project we extended the core MAP-Elites algorithm to solve continuous constrained optimization problem. Specifically, the problem setting is defined by an objective function subject to some constraints, some examples [here](https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_constrained_optimization).

**TODO**: Extend this part

## Reference

Mouret, J. B., & Clune, J. (2015). Illuminating search spaces by mapping elites. arXiv preprint arXiv:1504.04909. [[http://arxiv.org/abs/1504.04909](http://arxiv.org/abs/1504.04909)]