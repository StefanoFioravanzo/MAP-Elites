from math import sin, cos, e
import operator


class ConstrainedFunctions:

    def __init__(self, optima):
        self.optima = optima


class Rosenbrok:

    @staticmethod
    def evaluate(X):
        """
        http://en.wikipedia.org/wiki/Rosenbrock_function
        """
        x = X[0]
        y = X[1]
        a = 1. - x
        b = y - x * x
        return a * a + b * b * 100.

    @staticmethod
    def constraints():
        """
        http://en.wikipedia.org/wiki/Test_functions_for_optimization
        """
        return {
            "cubic":
                {
                    "fun": lambda x: (x[0] - 1.) ** 3. - x[1] + 1.,
                    "op": operator.le,
                    "target": 0
                },
            "line":
                {
                    "fun": lambda x: x[0] + x[1] - 2.,
                    "op": operator.le,
                    "target": 0
                }
        }


class MishraBird:

    @staticmethod
    def evaluate(X):
        x = X[0]
        y = X[1]
        a = 1. - cos(x)
        b = 1. - sin(y)
        c = x - y
        return sin(y) * e ** (a * a) + cos(x) * e ** (b * b) + c * c

    @staticmethod
    def constraints():
        """
        http://en.wikipedia.org/wiki/Test_functions_for_optimization
        """
        return (
            {
                "type": "circle",
                "func": lambda x: (x[0] + 5) ** 2 + (x[1] + 5) ** 2,
                "op": operator.lt,
                "target": 25
            })
