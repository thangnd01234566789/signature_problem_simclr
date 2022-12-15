import numpy as np

np.random.seed(0)

class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key"""

    def __init__(self, base_transform, n_view = 2):
        self.base_transform = base_transform
        self.n_view = n_view

    def __call__(self, x):
        return [self.base_transform(x) for b in range(self.n_view)]