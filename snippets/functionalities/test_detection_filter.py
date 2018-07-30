import numpy as np

arr = np.array([[[[0., 1., 0.4233, 435., 435., 345., 234.],
                  [0., 2., 0., 0., 0., 0., 0.],
                  [0., 1., 0.3423, 435., 212., 364., 553.],
                  [0., 4., 0.3562, 356., 221., 775., 568.]]]])
# sliced = arr[0, 0, :, 1:].copy()
sliced = arr[0, 0, :, 1:]
# np.copyto(sliced, arr[0, 0, :, 1:])
sliced = sliced[np.all(sliced != 0, axis=1)]
print(sliced)
deleted = np.delete(sliced, np.where(sliced[]))
# inserted = np.insert(sliced, 0, 0., axis=1)
# print(inserted)


# arr[0, 0, :, 1:] = arr[0, 0, :, 1:](np.all(arr[0, 0, :, 1:] != 0, axis=1))
# print(arr)

