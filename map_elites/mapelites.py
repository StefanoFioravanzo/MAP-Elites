from abc import ABC, abstractmethod

import numpy as np
import logging
from tqdm import tqdm

from .ea_operators import EaOperators
from .plot_utils import plot_heatmap_2d, plot_heatmap_2d_seaborn


class FeatureDimension:
    """
    Describes a feature dimension
    """

    def __init__(self, name, feature_function, bins):
        """
        :param name: Name or description of the feature dimension
        :param feature_function: Feature function or simulation, given a candidate solution as input
        :param bins: Array of bins, from starting value to last value of last bin
        """
        self.name = name
        # TODO: check feature function is indeed a function
        self.feature_function = feature_function
        self.bins = bins

    def feature_descriptor(self, x):
        """
        Simulate the candidate solution x and record its feature descriptor
        :param x: genotype of candidate solution x
        :return:
        """
        # TODO: what to do besides just calling the function?
        return self.feature_function(x)

    def discretize(self, value):
        """
        Get bin (index) of dimension from real value
        """
        index = np.digitize([value], self.bins)[0]
        if index in [0, len(self.bins)]:
            raise Exception(f"Constraint {self.name}: value {value} outside of bins {self.bins}")
        # - 1 because digitize is 1-indexed
        return index - 1


class MapElites(ABC):

    def __init__(self,
                 iterations,
                 random_solutions,
                 mutation_op,
                 mutation_args,
                 crossover_op,
                 crossover_args,
                 notebook=False):
        """
        :param iterations:
        :param random_solutions:
        :param notebook: True the code is executed inside an IPython Notebook.
        """
        self.notebook = notebook

        self.iterations = iterations
        self.random_solutions = random_solutions

        self.mutation_op = mutation_op
        self.mutation_args = mutation_args
        self.crossover_op = crossover_op
        self.crossover_args = crossover_args

        self.feature_dimensions = self.generate_feature_dimensions()
        # Check feature dimensions were initialized properly
        if not isinstance(self.feature_dimensions, (list, tuple)) or \
                not all(isinstance(ft, FeatureDimension) for ft in self.feature_dimensions):
            raise Exception(
                f"MapElites: `feature_dimensions` must be either a list or a tuple "
                f"object of {FeatureDimension.__name__} objects")

        # get number of bins for each feature dimension
        ft_bins = [len(ft.bins) - 1 for ft in self.feature_dimensions]

        # Map of Elites: Initialize data structures to store solutions and fitness values
        self.solutions = np.full(ft_bins, (np.inf, np.inf), dtype=(float, 2))
        self.performances = np.full(ft_bins, np.inf)

        logging.info("Configuration completed.")

    @staticmethod
    def from_config(cls, config):
        iterations = config['mapelites'].getint('iterations')
        random_solutions = config['mapelites'].getint('initial_random_population')
        notebook = config['env'].getboolean('notebook')

        # get list of ea operators
        ea_operators = [func for func in dir(EaOperators)
                        if callable(getattr(EaOperators, func))
                        and not func.startswith("__", 0, 2)
                        ]

        # get mutation and selection operators
        mutation_op = config['mutation']['type']
        mutation_fun = f"{str.lower(mutation_op)}_mutation"
        if mutation_fun not in ea_operators:
            raise ValueError(f"Mutation operator {mutation_op} not implemented.")
        mutation_fun = getattr(EaOperators, mutation_fun)
        mutation_args = None
        if mutation_op == "GAUSSIAN":
            mutation_args = {
                "mu": config['mutation'].getfloat('mu'),
                "sigma": config['mutation'].getfloat('sigma'),
                "indpb": config['mutation'].getfloat('indpb')
            }

        crossover_op = config['crossover']['type']
        crossover_fun = f"{str.lower(crossover_op)}_crossover"
        if crossover_fun not in ea_operators:
            raise ValueError(f"Crossover operator {crossover_op} not implemented.")
        crossover_fun = getattr(EaOperators, crossover_fun)
        crossover_args = None
        if crossover_op == "UNIFORM":
            crossover_args = {
                "indpb": config['crossover'].getfloat('indpb')
            }

        return cls(
            iterations=iterations,
            random_solutions=random_solutions,
            mutation_op=mutation_fun,
            mutation_args=mutation_args,
            crossover_op=crossover_fun,
            crossover_args=crossover_args,
            notebook=notebook
        )

    def generate_initial_population(self):
        logging.info("Generate initial population")
        # G the number of initial random solutions
        for _ in range(0, self.random_solutions):
            x = self.generate_random_solution()
            # add solution to elites computing features and performance
            self.place_in_mapelites(x)

        # self.plot_map_of_elites()

    def run(self):

        # start by creating an initial set of random solutions
        self.generate_initial_population()

        with tqdm(total=self.iterations, desc="Iterations completed") as pbar:
            for i in range(0, self.iterations):
                logging.info(f"ITERATION {i}")
                if self.stopping_criteria():
                    break

                # possible solution
                x = None
                logging.info("Select and mutate.")
                # get the index of a random individual
                ind = self.random_selection(individuals=1)[0]

                # TODO: random variation (+check for validity)
                ind = self.mutation_op(ind, **self.mutation_args)[0]
                self.place_in_mapelites(ind, pbar=pbar)

        # save results, display metrics and plot statistics
        self.plot_map_of_elites()

    def place_in_mapelites(self, x, pbar=None):
        """
        Puts a solution inside the N-dimensional map of elites space.
        The following criteria is used:
            - Compute the feature descriptor of the solution to find the correct
                cell in the N-dimensional space
            - Compute the performance of the solution
            - Check if the cell is empty of the previous performance is worse
                -> place new solution in the cell
        :param x: genotype of a solution
        """
        b = self.map_x_to_b(x)
        perf = self.performance_measure(x)

        # TODO: What about the case when performances are equal?
        if self.performances[b] > perf:
            logging.info(f"PLACE: Placing individual {x} at {b} with perf: {perf}")
            self.performances[b] = perf
            self.solutions[b] = x
        else:
            logging.info(f"PLACE: Individual {x} rejected at {b} with perf: {perf} in favor of {self.performances[b]}")
        if pbar is not None:
            pbar.update(1)

    def random_selection(self, individuals=1):
        """
        Select an elite x from the current map of elites
        The selection is done by selecting a random bin for each feature
        dimension, until a bin with a value is found.
        # TODO: To optimize this procedure we could keep a data structure that records the cells that have been filled
        :param individuals: The number of individuals to randomly select
        :return: A list of N random elites
        """

        def _get_random_index():
            indexes = tuple()
            for ft in self.feature_dimensions:
                rnd_ind = np.random.randint(0, len(ft.bins) - 1, 1)[0]
                indexes = indexes + (rnd_ind,)
            return indexes

        def _is_not_initialized(index):
            """
            Checks if the selected index points to a NaN or Inf solution (not yet initialized)
            The solution is considered as NaN/Inf if any of the dimensions of the individual is NaN/Inf
            :return:
            """
            return any([x == np.nan or np.abs(x) == np.inf for x in self.solutions[index]])

        # individuals
        inds = list()
        idxs = list()
        for _ in range(0, individuals):
            idx = _get_random_index()
            # we do not want to repeat entries
            while idx in idxs or _is_not_initialized(idx):
                idx = _get_random_index()
            idxs.append(idx)
            inds.append(self.solutions[idx])
        return inds

    def plot_map_of_elites(self):
        """
        Plot a heatmap of elites
        """
        def stringify_bin_axis(bins):
            """
            Takes as input an array of values describing the bins of the feature dimension.
            We want to have as output an array of strings in the form of "from _start_value to _end_value"
            that describe the range of each bin.
            To do this we zip the bins array to itself (shifting one of the two by one element) to
            produce the string. The take all the resulting strings but the first (which is the additional one
            produced due to shifting)
            """
            return list(map(lambda x: f'{x[0]}: {x[1]} to {x[2]}', zip(range(0, len(bins)), np.insert(bins, 0, 0), bins)))[1:]

        # Prepare data for plotting
        values = np.reshape(self.performances, (-1, ))

        x_ax = stringify_bin_axis(self.feature_dimensions[0].bins)
        y_ax = stringify_bin_axis(self.feature_dimensions[1].bins)
        # x_ax = list(map(lambda x: f"{x}", self.feature_dimensions[0].bins[1:]))
        # y_ax = list(map(lambda x: f"{x}", self.feature_dimensions[1].bins[1:]))
        data = np.stack([np.repeat(x_ax, len(y_ax)), np.tile(y_ax, len(x_ax)), values], axis=1)

        plot_heatmap_2d_seaborn(data, x_ax, y_ax, "X", "Y", notebook=self.notebook)

    def stopping_criteria(self):
        """
        Any criteria to stop the simulation before the given number of runs
        :return: True if the algorithm has to stop. False otherwise.
        """
        return False

    def validate_config(self):
        """
        Validate configuration file
        """
        raise NotImplementedError

    @abstractmethod
    def performance_measure(self, x):
        """
        Function to evaluate solution x and give a performance measure
        :param x: genotype of a solution
        :return: performance measure of that solution
        """
        pass

    @abstractmethod
    def map_x_to_b(self, x):
        """
        Function to map a solution x to feature space dimensions
        :param x: genotype of a solution
        :return: phenotype of the solution (tuple of indices of the N-dimensional space)
        """
        pass

    @abstractmethod
    def generate_random_solution(self):
        """
        Function to generate an initial random solution x
        :return: x, a random solution
        """
        pass

    @abstractmethod
    def generate_feature_dimensions(self):
        """
        Generate a list of FeatureDimension objects to define the feature dimension functions
        :return: List of FeatureDimension objects
        """
        pass