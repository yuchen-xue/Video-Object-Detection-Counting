import numpy as np

arr = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6]])
print(arr.shape)
for i, s in enumerate(arr):
    if i == 2:
        # arr = np.delete(arr, np.where(arr == s), 0)
        arr = np.delete(arr, np.where(np.all(arr == s, axis=1)), 0)
    if i == 3:
        arr = np.delete(arr, np.where(np.all(arr == s, axis=1)), 0)
    if i == 4:
        # arr = np.delete(arr, i, 0)
        arr = np.delete(arr, np.where(np.all(arr == s, axis=1)), 0)
print(arr)

arr = np.delete(arr, np.where(np.all(arr == [3, 3, 3, 3], axis=1)), 0)
print(arr)
