import numpy as np
a = np.random.random((8, 5))
print("the matrix size is:", a.shape)
u = np.linalg.svd(a, False)
print(u[0])
print(u[1])


