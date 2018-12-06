#!/usr/bin/env python
# coding: utf-8

import logging
import configparser
import numpy as np

from map_elites.mapelites import MapElites, FeatureDimension

logging.basicConfig(filename="log.log", level=logging.INFO)
# this should set the seed project wide
np.random.seed(1)
config = configparser.ConfigParser()

# TODO: Implement proper logging at each iteration to check for issues
# es: iter1 - perf - ft_desc - etc...
class MapElitesContinuousOpt(MapElites):

    def __init__(self, *args, **kwargs):
        super(MapElitesContinuousOpt, self).__init__(*args, **kwargs)

    def map_x_to_b(self, x):
        """
        Map X solution to feature space dimension, meaning:
            - apply the constraints to a solution
        :return: tuple of indexes
        """
        b = tuple()
        for ft in self.feature_dimensions:
            desc = ft.feature_descriptor(x)
            i = ft.discretize(desc)
            b = b + (i,)

        return b

    def performance_measure(self, x):
        """
        Apply the fitness continuous function to x
        """
        logging.info("calculate performance measure")
        return self.F.evaluate(x)

    # TODO: Ask the professor about this
    def generate_random_solution(self):
        """
        To ease the bootstrap of the algorithm, we can generate
        the first solutions in the feature space, so that we start
        filling the bins
        """
        logging.info("Generate random solution")

        dimensions = self.F.get_domain()
        return np.array([
            np.random.uniform(d[0], d[1], 1)[0] for d in dimensions
        ])

    def generate_feature_dimensions(self):
        default_bins = [-np.inf, 0.0, 4.0, 6.0, 8.0, 10.0, np.inf]

        return [
            FeatureDimension(name=v['name'],
                             feature_function_target=v['target'],
                             feature_function_call=v['func'],
                             feature_function_operator=v['op'],
                             bins=default_bins)
            for k, v in self.F.constraints().items()
        ]


# read configuration file
config.read('config.ini')

logging.info("Start map elites")
map_E = MapElitesContinuousOpt.from_config(MapElitesContinuousOpt, config)
map_E.run()
