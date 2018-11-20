from map_elites.mapelites import FeatureDimension
from mapelites_continuous_opt import MapElitesContinuousOpt
from functions import Rosenbrok

import numpy as np


def main():
    rosenbrok_const = Rosenbrok.constraints()

    cubic_bins = [-np.inf, 0.0, 4.0, 4.1, 4.2, 4.3, np.inf]
    cubic = FeatureDimension("cubic_function", rosenbrok_const['cubic']['fun'], cubic_bins)

    line_bins = [-np.inf, 0.0, 3.0, 3.1, 3.2, 3.3, np.inf]
    line = FeatureDimension("line_function", rosenbrok_const['line']['fun'], line_bins)

    ft_dimensions = [cubic, line]

    iterations = 10
    initial_population = 4
    map_E = MapElitesContinuousOpt(
        iterations=iterations,
        random_solutions=initial_population,
        feature_dimensions=ft_dimensions
    )
    map_E.run()


if __name__ == "__main__":
    # TODO: Use config file (es .ini) to configure parameters of experiment
    # Es: type of fitness function, the constraints or the parameters of the constraints.
    main()