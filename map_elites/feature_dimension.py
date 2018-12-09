import math
import operator

import numpy as np


class FeatureDimension:
    """
    Implements a feature dimension of the MAP-Elites algorithm
    """

    def __init__(self, name, feature_function_target, feature_function_call, feature_function_operator, bins):
        """
        :param name: Name of the feature dimension
        :param feature_function_call: The function to compute or simulate the feature dimension value
        :param feature_function_target: Value target of the feature simulation
        :param feature_function_operator: Operator used to compare the simulated feature value with the target
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
                # return negative number instead of 0 because the discretize() function includes
                # the lower bound value into the following bin. So 0 would result in the second bin.
                return -.1
            else:
                return math.fabs(self.feature_function_call(x) - self.feature_function_target(x))

    def discretize(self, value):
        """
        Get bin (index) of dimension from real value
        """
        index = np.digitize([value], self.bins)[0]
        if index in [0, len(self.bins)]:
            raise Exception(f"Constraint {self.name}: value {value} outside of bins {self.bins}")
        # - 1 because digitize is 1-indexed
        return index - 1
