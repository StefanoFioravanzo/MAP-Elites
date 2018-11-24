import random

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
    def gaussian_mutation(individual, mu, sigma, indpb):
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
        size = len(individual)
        if not isinstance(mu, Sequence):
            mu = repeat(mu, size)
        elif len(mu) < size:
            raise IndexError(f"mu must be at least the size of individual: {len(mu)} < {size}")
        if not isinstance(sigma, Sequence):
            sigma = repeat(sigma, size)
        elif len(sigma) < size:
            raise IndexError(f"sigma must be at least the size of individual: {len(sigma)} < {size}")

        for i, m, s in zip(range(size), mu, sigma):
            if random.random() < indpb:
                individual[i] += random.gauss(m, s)

        return individual,
