import numpy as np

class Constant:
    R = 1
    Q = np.array([[100, 100, 0, 0],
                  [100, 100, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])
    alpha = 3
    lamb = 0.01