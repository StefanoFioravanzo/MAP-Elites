import logging
import operator
import configparser

import numpy as np

from tqdm import tqdm
from pathlib import Path
from shutil import copyfile
from datetime import datetime
from abc import ABC, abstractmethod

# local imports
import functions
from .feature_dimension import FeatureDimension
from .plot_utils import plot_heatmap
from .ea_operators import EaOperators


class MapElites(ABC):

    def __init__(self,
                 iterations,
                 optimization_function,
                 optimization_function_dimensions,
                 bootstrap_individuals,
                 mutation_op,
                 mutation_args,
                 crossover_op,
                 crossover_args,
                 bins,
                 plot_args,
                 log_dir,
                 overwrite_log_dir,
                 config_path,
                 minimization=True,
                 ):
        """
        :param iterations: Number of evolutionary iterations
        :param optimization_function: The function to be optimized
        :param optimization_function_dimensions: The number of dimensions of the function to be optimized
        :param bootstrap_individuals: Number of individuals randomly generated to bootstrap the algorithm
        :param mutation_op: Mutation function
        :param mutation_args: Mutation function arguments
        :param crossover_op: Crossover function
        :param crossover_args: Crossover function arguments
        :param bins: Bins for feature dimensions
        :param minimization: True if solving a minimization problem. False if solving a maximization problem.
        """
        self.minimization = minimization
        # set the choice operator either to do a minimization or a maximization
        if self.minimization:
            self.place_operator = operator.lt
        else:
            self.place_operator = operator.ge

        self.plot_args = plot_args

        self.F = optimization_function(optimization_function_dimensions)
        self.iterations = iterations
        self.random_solutions = bootstrap_individuals
        self.bins = bins

        self.mutation_op = mutation_op
        self.mutation_args = mutation_args
        # add to the mutation args the boundaries (domain) of the optimization function
        self.mutation_args['boundaries'] = self.F.get_domain()
        self.crossover_op = crossover_op
        self.crossover_args = crossover_args

        self.feature_dimensions = self.generate_feature_dimensions()
        # Check feature dimensions were initialized properly
        if not isinstance(self.feature_dimensions, (list, tuple)) or \
                not all(isinstance(ft, FeatureDimension) for ft in self.feature_dimensions):
            raise Exception(
                f"MapElites: `feature_dimensions` must be either a list or a tuple "
                f"object of { FeatureDimension.__name__} objects")

        # get number of bins for each feature dimension
        ft_bins = [len(ft.bins) - 1 for ft in self.feature_dimensions]

        # Map of Elites: Initialize data structures to store solutions and fitness values
        self.solutions = np.full(
            ft_bins, (np.inf, ) * optimization_function_dimensions,
            dtype=(float, optimization_function_dimensions)
        )
        self.performances = np.full(ft_bins, np.inf)

        # configure logging
        # empty log file from potential previous logs
        open('log.log', 'w').close()

        if log_dir:
            self.log_dir_path = Path(log_dir)
        else:
            now = datetime.now().strftime("%Y%m%d%H%M%S")
            self.log_dir_name = f"log_{now}"
            self.log_dir_path = Path(f'logs/{self.log_dir_name}')
        # create log dir
        self.log_dir_path.mkdir(parents=True, exist_ok=overwrite_log_dir)
        # save config file
        copyfile(config_path, self.log_dir_path / 'config.ini')

        logging.basicConfig(filename=self.log_dir_path / "log.log", level=logging.INFO)
        logging.info("Configuration completed.")

    @classmethod
    def from_config(cls, config_path, log_dir=None, overwrite=False):
        """
        Read config file and create a MAP-Elites instance.
        :param config_path: Path to config.ini file
        :param log_dir: Absolute path to logging directory
        """
        # Read configuration file
        config = configparser.ConfigParser()
        config.read(config_path)

        # RANDOM SEED
        seed = config['mapelites'].getint('seed')
        np.random.seed(seed)

        # MAIN MAPELITES CONF
        iterations = config['mapelites'].getint('iterations')
        bootstrap_individuals = config['mapelites'].getint('bootstrap_individuals')
        minimization = config['mapelites'].getboolean('minimization')

        # PLOTTING CONF
        plot_args = dict()
        plot_args['highlight_best'] = config['plotting'].getboolean('highlight_best')

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
        mutation_boundary_management = config['mutation']['boundary']

        mutation_boundary_values = ['saturation', 'bounce', 'toroidal']
        if mutation_boundary_management not in mutation_boundary_values:
            raise ValueError(f"The mutation boundary management must be one of {mutation_boundary_values}")

        mutation_args = None
        if mutation_op == "GAUSSIAN":
            mutation_args = {
                "mu": config['mutation'].getfloat('mu'),
                "sigma": config['mutation'].getfloat('sigma'),
                "indpb": config['mutation'].getfloat('indpb'),
                "boundary_management": mutation_boundary_management
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
            bootstrap_individuals=bootstrap_individuals,
            mutation_op=mutation_fun,
            mutation_args=mutation_args,
            crossover_op=crossover_fun,
            crossover_args=crossover_args,
            minimization=minimization,
            plot_args=plot_args,
            log_dir=log_dir,
            config_path=config_path,
            overwrite_log_dir=overwrite,
            bins=bins
        )

    def generate_initial_population(self):
        """
        Bootstrap the algorithm by generating `self.bootstrap_individuals` individuals
        randomly sampled from a uniform distribution
        """
        logging.info("Generate initial population")
        for _ in range(0, self.random_solutions):
            x = self.generate_random_solution()
            # add solution to elites computing features and performance
            self.place_in_mapelites(x)

    def run(self):
        """
        Main iteration loop of MAP-Elites
        """
        # start by creating an initial set of random solutions
        self.generate_initial_population()

        # tqdm: progress bar
        with tqdm(total=self.iterations, desc="Iterations completed") as pbar:
            for i in range(0, self.iterations):
                logging.info(f"ITERATION {i}")
                if self.stopping_criteria():
                    break

                logging.info("Select and mutate.")
                # get the index of a random individual from the map of elites
                ind = self.random_selection(individuals=1)[0]
                # mutate the individual
                ind = self.mutation_op(ind, **self.mutation_args)[0]
                # place the new individual in the map of elites
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
        - Check if the cell is empty or if the previous performance is worse
            - Place new solution in the cell
        :param x: genotype of an individual
        :param pbar: TQDM progress bar instance
        """
        # get coordinates in the feature space
        b = self.map_x_to_b(x)
        # performance of the optimization function
        perf = self.performance_measure(x)
        # place operator performs either minimization or maximization
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
        Select an elite x from the current map of elites.
        The selection is done by selecting a random bin for each feature
        dimension, until a bin with a value is found.
        :param individuals: The number of individuals to randomly select
        :return: A list of N random elites
        """

        def _get_random_index():
            """
            Get a random cell in the N-dimensional feature space
            :return: N-dimensional tuple of integers
            """
            indexes = tuple()
            for ft in self.feature_dimensions:
                rnd_ind = np.random.randint(0, len(ft.bins) - 1, 1)[0]
                indexes = indexes + (rnd_ind,)
            return indexes

        def _is_not_initialized(index):
            """
            Checks if the selected index points to a NaN or Inf solution (not yet initialized)
            The solution is considered as NaN/Inf if any of the dimensions of the individual is NaN/Inf
            :return: Boolean
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
        """
        Save logs, config file and data structures to log folder
        """
        np.save(self.log_dir_path / 'performances', self.performances)
        np.save(self.log_dir_path / "solutions", self.solutions)

    def plot_map_of_elites(self):
        """
        Plot a heatmap of elites
        """
        # Stringify the bins to be used as strings in the plot axes
        if len(self.feature_dimensions) == 1:
            y_ax = ["-"]
            x_ax = [str(d) for d in self.feature_dimensions[0].bins]
        else:
            x_ax = [str(d) for d in self.feature_dimensions[0].bins]
            y_ax = [str(d) for d in self.feature_dimensions[1].bins]

        plot_heatmap(self.performances,
                     x_ax,
                     y_ax,
                     savefig_path=self.log_dir_path,
                     title=f"{self.F.__class__.__name__} function",
                     **self.plot_args)

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
