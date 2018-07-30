import numpy as np
idx = 8
arr = np.arange(idx*7).reshape((1, 1, idx, 7))
print(arr)
# sliced = arr[0][0][:, 0:2][:]
sliced = arr[0, 0, :, :2]

print(sliced)
print(sliced.shape)
