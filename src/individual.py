import numpy as np

class Individual:
    def __init__(self, features):
        self.features = np.array(features)
        self.objectives = np.array([0.0, 0.0])
        self.accuracy = 0.0