from abc import ABC, abstractmethod

from math import sin, cos, e, atan2, atan, sqrt, pi
import operator
import numpy as np


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
            "g1":
                {
                    "name": "g1",
                    "func": lambda x: (x[0] - 1.) ** 3. - x[1] + 1.,
                    "op": operator.le,
                    "target": lambda x: 0
                },
            "g2":
                {
                    "name": "g2",
                    "func": lambda x: x[0] + x[1] - 2.,
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
            "g1":
                {
                    "name": "g1",
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
        b = cos(a) ** 2
        c = 3 * x + y
        d = x * sin(c)
        return (b * (-1.)) - d

    def constraints(self):

        def const1(X):
            x = X[0]
            y = X[1]
            t = atan2(x, y)
            a = (2 * cos(t)) - (0.5 * cos(2 * t)) - (0.25 * cos(3 * t)) - (0.125 * cos(4 * t))
            return a ** 2 + (2 * sin(t)) ** 2

        return {
            "g1":
                {
                    "name": "g1",
                    "func": const1,
                    "op": operator.gt,
                    "target": lambda x: x[0] ** 2 + x[1] ** 2
                }
        }

    def get_domain(self):
        return [
            (-2.25, 2.5),
            (-2.5, 1.75)
        ]


# TODO: Crea numeri complessi?
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
            "g1":
                {
                    "name": "g1",
                    "func": const1,
                    "op": operator.ge,
                    "target": lambda x: x[0] ** 2 + x[1] ** 2
                }
        }

    def get_domain(self):
        return [
            (-1.25, 1.25),
            (-1.25, 1.25)
        ]


class C16(ConstrainedFunction):

    def __init__(self, dimensions):
        self.o = np.array(
            [0.365972807627352, 0.429881383400138, -0.420917679577772, 0.984265986788929, 0.324792771198785,
             0.463737106835568, 0.989554882052943, 0.307453878359996, 0.625094764380575, -0.358589007202526,
             0.24624504504104, -0.96149609569083, -0.184146201911073, -0.030609388103067, 0.13366054512765,
             0.450280168292005, -0.662063233352676, 0.720384516339946, 0.518473305175091, -0.969074121149791,
             -0.221655317677079, 0.327361832246864, -0.695097713581401, -0.671724285177815, -0.534907819936839,
             -0.003991036739113, 0.486452090756303, -0.689962754053575, -0.138437260109118, -0.626943354458217])

        if dimensions > len(self.o):
            raise ValueError("Dimensions cannot be higher than o vector")

        super().__init__(dimensions)

    def evaluate(self, X):
        x = np.array(X)
        z = x - self.o[:self.D]
        a = np.sum([(_z ** 2) / 4000] for _z in z)
        b = np.prod([cos(_z / sqrt(i)) for i, _z in enumerate(z)])
        return a - b + 1

    def constraints(self):

        def g1(X):
            x = np.array(X)
            z = x - self.o[:self.D]
            return np.sum([_z * _z - 100 * cos(pi * _z) + 10 for _z in z])

        def g2(X):
            x = np.array(X)
            z = x - self.o[:self.D]
            return np.prod(z[:self.D])

        def h1(X):
            x = np.array(X)
            z = x - self.o[:self.D]
            return np.sum([_z * sin(sqrt(abs(_z))) for _z in z])

        def h2(X):
            x = np.array(X)
            z = x - self.o[:self.D]
            return np.sum([_z * -1. * sin(sqrt(abs(_z))) for _z in z])

        return {
            "g1":
                {
                    "name": "g1",
                    "func": g1,
                    "op": operator.le,
                    "target": lambda x: 0
                },
            "g2":
                {
                    "name": "g2",
                    "func": g2,
                    "op": operator.le,
                    "target": lambda x: 0
                },
            "h1":
                {
                    "name": "h1",
                    "func": h1,
                    "op": operator.eq,
                    "target": lambda x: 0
                },
            "h2":
                {
                    "name": "h2",
                    "func": h2,
                    "op": operator.eq,
                    "target": lambda x: 0
                }
        }

    def get_domain(self):
        return [
            (-10, 10) for _ in range(0, self.D)
        ]
