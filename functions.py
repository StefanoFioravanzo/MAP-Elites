from abc import ABC, abstractmethod

from math import sin, cos, e, atan2, atan
import operator


class ConstrainedFunction(ABC):

    def __init__(self, dimensions):
        self.D = dimensions

    @abstractmethod
    def evaluate(self, X):
        pass

    @abstractmethod
    def constraints(self):
        pass

    @abstractmethod
    def get_domain(self):
        """
        Return a list of tuple representing a domain
        of admissible individuals for each dimension
        """
        pass


class Rosenbrok(ConstrainedFunction):

    def evaluate(self, X):
        """
        http://en.wikipedia.org/wiki/Rosenbrock_function
        """
        x = X[0]
        y = X[1]
        a = 1. - x
        b = y - x * x
        return a * a + b * b * 100.

    def constraints(self):
        """
        http://en.wikipedia.org/wiki/Test_functions_for_optimization
        """
        return {
            "cubic":
                {
                    "name": "cubic_function",
                    "fun": lambda x: (x[0] - 1.) ** 3. - x[1] + 1.,
                    "op": operator.le,
                    "target": lambda x: 0
                },
            "line":
                {
                    "name": "cubic_function",
                    "fun": lambda x: x[0] + x[1] - 2.,
                    "op": operator.le,
                    "target": lambda x: 0
                }
        }

    def get_domain(self):
        return [
            (-1.5, 1.5),
            (-0.5, 2.5)
        ]


class MishraBird(ConstrainedFunction):

    def evaluate(self, X):
        x = X[0]
        y = X[1]
        a = 1. - cos(x)
        b = 1. - sin(y)
        c = x - y
        return sin(y) * e ** (a * a) + cos(x) * e ** (b * b) + c * c

    def constraints(self):
        """
        http://en.wikipedia.org/wiki/Test_functions_for_optimization
        """
        return {
            "circle":
                {
                    "name": "circle_function",
                    "func": lambda x: (x[0] + 5) ** 2 + (x[1] + 5) ** 2,
                    "op": operator.lt,
                    "target": lambda x: 0
                }
        }

    def get_domain(self):
        return [
            (-10, 0),
            (-6.5, 0)
        ]


class Townsend(ConstrainedFunction):

    def evaluate(self, X):
        x = X[0]
        y = X[1]
        a = (x - 0.1) * y
        b = cos(a) ** cos(a)
        c = 3 * x + y
        d = x * sin(c)
        return (b * (-1.)) - d

    def constraints(self):

        def const1(X):
            x = X[0]
            y = X[1]
            t = atan2(x, y)
            a = (2 * cos(t)) - (0.5 * cos(2 * t)) - (0.25 * cos(3 * t)) - (0.125 * cos(4 * t))
            return a ** a + (2 * sin(t)) ** (2 * sin(t))

        return {
            "const1":
                {
                    "name": "const1",
                    "func": const1,
                    "op": operator.gt,
                    "target": lambda x: x[0] ** x[0] + x[1] ** x[1]
                }
        }

    def get_domain(self):
        return [
            (-2.25, 2.5),
            (-2.5, 1.75)
        ]


class Simionescu(ConstrainedFunction):

    def evaluate(self, X):
        x = X[0]
        y = X[1]
        return 0.1 * x * y

    def constraints(self):

        def const1(X):
            x = X[0]
            y = X[1]
            rt = 1.
            rs = 0.2
            n = 8
            a = n * atan(x / y)
            b = rt + rs * a
            return b ** b

        return {
            "const1":
                {
                    "name": "const1",
                    "func": const1,
                    "op": operator.ge,
                    "target": lambda x : x[0] ** x[0] + x[1] ** x[1]
                }
        }

    def get_domain(self):
        return [
            (-1.25, 1.25),
            (-1.25, 1.25)
        ]

