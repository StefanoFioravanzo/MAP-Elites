from map_elites.mapelites import MapElites


class MapElitesContinuousOpt(MapElites):

    def __init__(self, *args, **kwargs):
        super(MapElitesContinuousOpt, self).__init__(*args, **kwargs)

    def map_x_to_b(self, x):
        """
        Map X solution to feature space dimension, meaning:
            - apply the constraints to a solution
        """

    def performance_measure(self, x):
        """
        Apply the fitness continuous function to x
        """
        pass

    def generate_random_solution(self):
        pass

    def init_feature_dimentions(self):
        pass