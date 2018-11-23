from abc import ABC, abstractmethod
import logging

import numpy as np

from ea_operators import gaussian_mutation


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
        print(x)
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

    def __init__(self, iterations, random_solutions, feature_dimensions):
        """
        :param iterations:
        :param random_solutions:
        :param feature_dimensions:
        """
        self.iterations = iterations
        self.random_solutions = random_solutions

        self.feature_dimensions = feature_dimensions
        # Check feature dimensions were initialized properly
        if not isinstance(self.feature_dimensions, (list, tuple)) or \
                not all(isinstance(ft, FeatureDimension) for ft in self.feature_dimensions):
            raise Exception(
                f"MapElites: `feature_dimensions` must be either a list or a tuple "
                f"object of {FeatureDimension.__name__} objects")

        # get number of bins for each feature dimension
        ft_bins = [len(ft.bins) - 1 for ft in self.feature_dimensions]

        # Map of Elites: Initialize data structures to store solutions and fitness values
        self.solutions = np.full(ft_bins, (-np.inf, -np.inf), dtype=(float, 2))
        self.performances = np.full(ft_bins, -np.inf)

        logging.info("Configuration completed.")

    def generate_initial_population(self):
        logging.info("Generate initial population")
        # G the number of initial random solutions
        for _ in range(0, self.random_solutions):
            x = self.generate_random_solution()
            # add solution to elites computing features and performance
            self.place_in_mapelites(x)

    def run(self):

        # start by creating an initial set of random solutions
        self.generate_initial_population()

        for i in range(0, self.iterations):
            logging.info(f"Iteration {i}")
            if self.stopping_criteria():
                break

            # possible solution
            x = None
            logging.info("Select and mutate.")
            # get the index of a random individual
            ind = self.random_selection(individuals=1)[0]

            # TODO: random variation (+check for validity)
            ind = gaussian_mutation(ind, 0, 1, 0.5)[0]
            self.place_in_mapelites(ind)

    def place_in_mapelites(self, x):
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
        logging.info("Place in MapElites")
        b = self.map_x_to_b(x)
        perf = self.performance_measure(x)

        # TODO: What about the case when performances are equal?
        if self.performances[b] < perf:
            self.performances[b] = perf
            self.solutions[b] = x

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

        def _is_nan(index):
            """
            Checks if the selected index points to a NaN solution (not yet initialized)
            The solution is considered as NaN if any of the dimensions of the individual is NaN
            :return:
            """
            return any([x == np.nan for x in self.solutions[index]])

        # individuals
        inds = list()
        idxs = list()
        for _ in range(0, individuals):
            idx = _get_random_index()
            # we do not want to repeat entries
            while idx in idxs or _is_nan(idx):
                idx = _get_random_index()
            idxs.append(idx)
            inds.append(self.solutions[idx])
        return inds

    def plot_map_of_elites(self, dimentions):
        """
        Plot a heatmap of elits
        :param dimentions:
        :return:
        """
        pass

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
        :return: x, a random slution
        """
        pass
