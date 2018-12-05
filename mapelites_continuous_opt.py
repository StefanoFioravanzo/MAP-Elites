#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging
import configparser
import numpy as np

from functools import reduce
import operator

from map_elites.mapelites import MapElites, FeatureDimension
from functions import Rosenbrok

logging.basicConfig(filename="log.log", level=logging.INFO)
# this should set the seed project wide
np.random.seed(1)
config = configparser.ConfigParser()


# In[2]:


class MapElitesContinuousOpt(MapElites):

    def __init__(self, *args, **kwargs):
        # function object instance to be optimized
        self.F = Rosenbrok(dimensions=2)
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
        rosenbrok_const = self.F.constraints()

        cubic_bins = [-np.inf, 0.0, 4.0, 6.0, 8.0, 10.0, np.inf]
        cubic = FeatureDimension("cubic_function", rosenbrok_const['cubic'], cubic_bins)

        line_bins = [-np.inf, 0.0, 4.0, 6.0, 8.0, 10.0, np.inf]
        line = FeatureDimension("line_function", rosenbrok_const['line'], line_bins)

        return [cubic, line]


# read configuration file
config.read('config.ini')

logging.info("Start map elites")
map_E = MapElitesContinuousOpt.from_config(MapElitesContinuousOpt, config)
map_E.run()


# In[ ]:




