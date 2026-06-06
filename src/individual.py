import numpy as np


class Individual:
    def __init__(self, features):
        self.features = np.array(features)
        # Dwa cele: f1 = liczba cech (min), f2 = -suma IV (min, bo maksymalizujemy IV)
        self.objectives = np.array([0.0, 0.0])

        self.rank = None
        self.crowding_distance = 0.0
        self.accuracy = 0.0

    def dominate(self, other):
        """
        Sprawdza czy ten osobnik dominuje 'other' w sensie Pareto.
        Uwaga: używane tylko pomocniczo/diagnostycznie — pymoo zarządza
        dominacją wewnętrznie podczas optymalizacji NSGA-II.
        """
        better_in_at_least_one = False
        for i in range(len(self.objectives)):
            if self.objectives[i] > other.objectives[i]:
                return False
            if self.objectives[i] < other.objectives[i]:
                better_in_at_least_one = True
        return better_in_at_least_one