import numpy as np


class Individual:
    def __init__(self, features):
        self.features = np.array(features)
        # Przywracamy 3 cele
        self.objectives = np.array([0.0, 0.0, 0.0])
        self.best_objectives = np.array([float('inf'), float('inf'), float('inf')])

        self.rank = None
        self.crowding_distance = 0.0
        self.accuracy = 0.0
        self.velocity = np.zeros(len(features), dtype=float)
        self.best_pos = np.copy(self.features)

    def dominate(self, other):
        better_in_at_least_one = False
        for i in range(len(self.objectives)):
            if self.objectives[i] > other.objectives[i]:
                return False
            if self.objectives[i] < other.objectives[i]:
                better_in_at_least_one = True
        return better_in_at_least_one
