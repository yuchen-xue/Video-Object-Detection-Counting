import numpy as np

# a = np.array([1, 1, 1, 1])
# b = np.array([2, 2, 2, 2])
# c = np.array([3, 3, 3, 3])
a = (1, 1, 1, 1)
b = (2, 2, 2, 2)
c = (3, 3, 3, 3)
arr = np.stack((a, b, c))
print(arr.shape)

for item in arr:
    print(tuple(item))

# arr = np.concatenate((arr, np.array([(4, 4, 4, 4)])))
arr = np.vstack((arr, np.array([(4, 4, 4, 4)])))
print(arr.shape)
