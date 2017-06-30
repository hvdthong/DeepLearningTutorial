import numpy as np


matrix = np.random.random([1024, 64])  # 64-dimensional embeddings
ids = np.array([0, 5, 17, 33])
print matrix[ids]  # prints a matrix of shape [4, 64]