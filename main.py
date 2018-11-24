import logging
import configparser
import numpy as np

from mapelites_continuous_opt import MapElitesContinuousOpt

logging.basicConfig(filename="log.log", level=logging.INFO)

# this should set the seed project wide
np.random.seed(1)

config = configparser.ConfigParser()


def main():
    # read configuration file
    config.read('config.ini')

    logging.info("Start map elites")
    map_E = MapElitesContinuousOpt.from_config(MapElitesContinuousOpt, config)
    map_E.run()


if __name__ == "__main__":
    main()
