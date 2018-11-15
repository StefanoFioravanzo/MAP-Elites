from abc import ABC, abstractmethod

import numpy as np


class FeatureDimension:
    """
    Describes a feature dimension
    """

    def __init__(self, name, feature_function, dtype, bins):
        """
        :param name: Name or description of the feature dimension
        :param feature_function: Feature function or simulation, given a candidate solution as input
        :param dtype: Data type of the resulting dimension (e.g. float, bool, ...)
        :param bins: Array of bins, from starting value to last value of last bin
        """
        self.name = name
        # TODO: check feature function is indeed a function
        self.feature_function = feature_function
        self.dtype = dtype
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
        index = np.digitize([value], self.bins)
        if index in [0, len(self.bins)]:
            raise Exception(f"Constraint {self.name}: value {value} outside of bins {self.bins}")
        return index


# TODO: Define as abstract class
class MapElites(ABC):

    def __init__(self, iterations, initial_population, feature_dimensions):
        self.iterations = iterations
        self.initial_population = initial_population
        # tuple with the length of each dimension
        self.dimensions = feature_dimensions
        # init ndarray with provided value
        self.solutions = np.full(feature_dimensions, -np.inf)
        self.performances = np.full(feature_dimensions, -np.inf)

    def generate_initial_population(self):
        # G the number of initial random solutions
        for i in self.initial_population:
            x = self.generate_random_solution()
            # add solution to elites computing features and performance
            self.place_in_mapelites(x)

    def run(self):
        for i in self.iter:
            # possible solution
            x = None
            # TODO: random selection (+check for validity)
            # TODO: random variation (+check for validity)
            self.place_in_mapelites(x)

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
        b = self.map_x_to_b(x)
        perf = self.performance_measure(x)

        # TODO: What about the case when performances are equal?
        if self.performances[b] < perf:
            self.performances[b] = perf
            self.solutions[b] = x

    def plot_map_of_elites(self, dimentions):
        """
        Plot a heatmap of elits
        :param dimentions:
        :return:
        """
        pass

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
