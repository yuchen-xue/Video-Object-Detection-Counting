import numpy as np

lst = [[1, 4, 5, 3], [2, 4, 2, 5], [3, 2, 1, 5]]
lst.append([3, 5, 3, 3])
arr = np.vstack(tuple(lst))
print(arr)
