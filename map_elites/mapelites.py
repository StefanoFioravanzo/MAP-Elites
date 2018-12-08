from abc import ABC, abstractmethod

import os
import math
import json
import logging
import operator
import numpy as np
import configparser
from tqdm import tqdm
from pathlib import Path
from shutil import copyfile
from datetime import datetime

import functions
from .plot_utils import plot_heatmap
from .ea_operators import EaOperators


class FeatureDimension:
    """
    Describes a feature dimension
    """

    def __init__(self, name, feature_function_target, feature_function_call, feature_function_operator, bins):
        """
        :param name: Name or description of the feature dimension
        :param feature_function: Feature function or simulation, given a candidate solution as input
        :param bins: Array of bins, from starting value to last value of last bin
        """
        self.name = name

        self.feature_function_call = feature_function_call
        self.feature_function_target = feature_function_target
        self.feature_function_operator = feature_function_operator

        if self.feature_function_operator not in [operator.eq, operator.le, operator.lt, operator.ge, operator.gt]:
            raise ValueError(f"Feature function operator not recognized")

        self.bins = bins

    def feature_descriptor(self, x):
        """
        Simulate the candidate solution x and record its feature descriptor
        :param x: genotype of candidate solution x
        :return: The amount of error from the feature descriptor bound
        """
        if self.feature_function_operator == operator.eq:
            return math.fabs(self.feature_function_call(x) - self.feature_function_target(x))
        else:
            if self.feature_function_operator(
                    self.feature_function_call(x),
                    self.feature_function_target(x)):
                return 0
            else:
                return math.fabs(self.feature_function_call(x) - self.feature_function_target(x))

    # TODO: Check reason the values of contraints explode after some iterations
    def discretize(self, value):
        """
        Get bin (index) of dimension from real value
        """
        index = np.digitize([value], self.bins)[0]
        if index in [0, len(self.bins)]:
            raise Exception(f"Constraint {self.name}: value {value} outside of bins {self.bins}")
        # - 1 because digitize is 1-indexed
        return index - 1

# TODO: Check why as the iterations increase the value inside the the map of elites increases as well
# TODO: Use class method for initialization
class MapElites(ABC):

    def __init__(self,
                 iterations,
                 optimization_function,
                 optimization_function_dimensions,
                 random_solutions,
                 mutation_op,
                 mutation_args,
                 crossover_op,
                 crossover_args,
                 bins,
                 minimization=True,
                 ):
        """
        :param iterations:
        :param random_solutions:
        :param minimization: True if solving a minimization problem. False if solving a maximization problem.
        """
        self.minimization = minimization
        # set the choice operator either to do a minimization or a maximization
        if self.minimization:
            self.place_operator = operator.lt
        else:
            self.place_operator = operator.ge

        self.F = optimization_function(optimization_function_dimensions)
        self.iterations = iterations
        self.random_solutions = random_solutions
        self.bins = bins

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

        # configure logging
        # empty log file from potential previous logs
        open('log.log', 'w').close()

        now = datetime.now().strftime("%Y%m%d%H%M%S")
        self.log_dir_name = f"log_{now}"
        self.log_dir_path = Path(f'logs/{self.log_dir_name}')
        # create log dir
        self.log_dir_path.mkdir(parents=True)

        logging.info("Configuration completed.")

    @staticmethod
    def from_config(cls, config_path):
        # Read configuration file
        config = configparser.ConfigParser()
        config.read(config_path)

        # RANDOM SEED
        seed = config['mapelites'].getint('seed')
        np.random.seed(seed)

        # MAIN MAPELITES CONF
        iterations = config['mapelites'].getint('iterations')
        random_solutions = config['mapelites'].getint('initial_random_population')
        minimization = config['mapelites'].getboolean('minimization')

        # OPTIMIZATION FUNCTION
        function_name = config['opt_function']['name']
        function_dimensions = config['opt_function'].getint('dimensions')
        function_class = getattr(functions, function_name)

        if not issubclass(function_class, functions.ConstrainedFunction):
            raise ValueError(
                f"Optimization function class {function_class.__name__} must be a "
                f"subclass of {functions.ConstrainedFunction.__name__}")

        # BINS
        d = dict(config.items('opt_function'))
        bins_names = filter(lambda s: s.startswith("bin"), d.keys())
        bins = {_k: d[_k] for _k in bins_names}

        # substitute strings "inf" at start and end of bins with -np.inf and np.inf
        for k, v in bins.items():
            b = v.split(',')
            inf_start = (b[0] == "inf")
            inf_end = (b[len(b)-1] == "inf")
            if inf_start:
                b.pop(0)
            if inf_end:
                b.pop(len(b)-1)
            # convert strings to floats
            b = list(map(float, b))
            # add back the inf values
            if inf_start:
                b.insert(0, -np.inf)
            if inf_end:
                b.insert(len(b), np.inf)
            bins[k] = b

        # EA OPERATORS
        ea_operators = [func for func in dir(EaOperators)
                        if callable(getattr(EaOperators, func))
                        and not func.startswith("__", 0, 2)
                        ]

        # MUTATION AND CROSSOVER OPS
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
            optimization_function=function_class,
            optimization_function_dimensions=function_dimensions,
            random_solutions=random_solutions,
            mutation_op=mutation_fun,
            mutation_args=mutation_args,
            crossover_op=crossover_fun,
            crossover_args=crossover_args,
            minimization=minimization,
            bins=bins
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
        self.save_logs()
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
        if perf == 0.:
            print(perf)
        if self.place_operator(perf, self.performances[b]):
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

    def save_logs(self):
        copyfile('log.log', self.log_dir_path / 'log.log')
        copyfile('config.ini', self.log_dir_path / 'config.ini')
        np.save(self.log_dir_path / 'performances', self.performances)
        np.save(self.log_dir_path / "solutions", self.solutions)

    def plot_map_of_elites(self):
        """
        Plot a heatmap of elites
        """
        if len(self.feature_dimensions) == 1:
            y_ax = ["-"]
            x_ax = [str(d) for d in self.feature_dimensions[0].bins]
        else:
            x_ax = [str(d) for d in self.feature_dimensions[0].bins]
            y_ax = [str(d) for d in self.feature_dimensions[1].bins]

        plot_heatmap(self.performances, x_ax, y_ax, savefig_path=self.log_dir_path)

    def stopping_criteria(self):
        """
        Any criteria to stop the simulation before the given number of runs
        :return: True if the algorithm has to stop. False otherwise.
        """
        return False

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