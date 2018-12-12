import random

from math import abs
from itertools import repeat
from collections import Sequence


class EaOperators:

    #################################################
    # CROSSOVER
    #################################################

    @staticmethod
    def uniform_crossover(ind1, ind2, indpb):
        """
        Executes a uniform crossover that modify in place the two
        individuals. The attributes are swapped according to the
        *indpb* probability.
        :param ind1: The first individual participating in the crossover.
        :param ind2: The second individual participating in the crossover.
        :param indpb: Independent probability for each attribute to be exchanged.
        :returns: A tuple of two individuals.
        """
        size = min(len(ind1), len(ind2))
        for i in range(size):
            if random.random() < indpb:
                ind1[i], ind2[i] = ind2[i], ind1[i]
        return ind1, ind2

    @staticmethod
    def one_point_crossover(ind1, ind2):
        """
        Executes a one point crossover on the input individuals.
        The two individuals are modified in place. The resulting individuals will
        respectively have the length of the other.
        :param ind1: The first individual participating in the crossover.
        :param ind2: The second individual participating in the crossover.
        :returns: A tuple of two individuals.
        """
        size = min(len(ind1), len(ind2))
        cxpoint = random.randint(1, size - 1)
        ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]

        return ind1, ind2

    #################################################
    # MUTATION
    #################################################

    @staticmethod
    def gaussian_mutation(individual, mu, sigma, indpb, boundary_management=None, boundaries=None):
        """
        This function applies a gaussian mutation of mean *mu* and standard
        deviation *sigma* on the input individual. This mutation expects a
        individual composed of real valued attributes.
        The *indpb* argument is the probability of each attribute to be mutated.
        :param individual: Individual to be mutated.
        :param mu: Mean or a sequence of means for the
                   gaussian addition mutation.
        :param sigma: Standard deviation or a sequence of
                      standard deviations for the gaussian addition mutation.
        :param indpb: Independent probability for each attribute to be mutated.
        :returns: A tuple of one individual.
        """

        def _default(x, delta, _):
            return x + delta

        def _saturation(x, delta, boundary):
            assert boundary[0] <= x <= boundary[1]
            if delta > 0 and x + delta > boundary[1]:
                return boundary[1]
            if delta < 0 and x + delta < boundary[0]:
                return boundary[0]
            return x + delta

        def _bounce(x, delta, boundary):
            assert boundary[0] <= x <= boundary[1]
            if delta > 0 and x + delta > boundary[1]:
                return boundary[1] - abs(x + delta - boundary[1])
            if delta < 0 and x + delta < boundary[0]:
                return boundary[0] + abs(x + delta - boundary[0])
            return x + delta

        def _toroidal(x, delta, boundary):
            assert boundary[0] <= x <= boundary[1]
            if delta > 0 and x + delta > boundary[1]:
                return boundary[0] + abs(x + delta - boundary[1])
            if delta < 0 and x + delta < boundary[0]:
                return boundary[1] - abs(x + delta - boundary[0])

        if boundaries:
            assert len(individual) == len(boundaries)
        bound_func = _default
        if boundary_management == "saturation":
            bound_func = _saturation
        if boundary_management == "bounce":
            bound_func = _bounce
        if boundary_management == "toroidal":
            bound_func = _toroidal

        size = len(individual)
        if not isinstance(mu, Sequence):
            mu = repeat(mu, size)
        elif len(mu) < size:
            raise IndexError(f"mu must be at least the size of individual: {len(mu)} < {size}")
        if not isinstance(sigma, Sequence):
            sigma = repeat(sigma, size)
        elif len(sigma) < size:
            raise IndexError(f"sigma must be at least the size of individual: {len(sigma)} < {size}")

        for i, m, s, b in zip(range(size), mu, sigma, boundaries):
            if random.random() < indpb:
                mut = random.gauss(m, s)
                individual[i] = bound_func(individual[i], mut, b)

        return individual,
