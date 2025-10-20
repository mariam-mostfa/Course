import numpy as np

A = np.array([10, 20, 30, 40, 50])
B = np.array([5, 4, 3, 2, 1])
print(A + B)
print(A - B)
print(A * B)
print(A / B)

print(np.mean(A))
print(A.max())
print(A.min())
print(np.dot(A, B))

reshaped = A.reshape(5, 1)
print(reshaped)
