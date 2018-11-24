import numpy as np
import logging

from map_elites.mapelites import FeatureDimension
from mapelites_continuous_opt import MapElitesContinuousOpt
from functions import Rosenbrok

logging.basicConfig(filename="log.log", level=logging.INFO)

# this should set the seed project wide
np.random.seed(1)


def main():
    rosenbrok_const = Rosenbrok.constraints()

    cubic_bins = [-np.inf, 0.0, 4.0, 6.0, 8.0, 10.0, np.inf]
    cubic = FeatureDimension("cubic_function", rosenbrok_const['cubic']['fun'], cubic_bins)

    line_bins = [-np.inf, 0.0, 4.0, 6.0, 8.0, 10.0, np.inf]
    line = FeatureDimension("line_function", rosenbrok_const['line']['fun'], line_bins)

    ft_dimensions = [cubic, line]

    iterations = 100
    initial_population = 100

    logging.info("Start map elites")
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