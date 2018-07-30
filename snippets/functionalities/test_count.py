import numpy as np

a = np.array([2., 0.12])
b = np.array([2., 0.34])
c = np.array([1., 0.48])
d = np.array([1., 0.])
arr = np.vstack((a, b, c, d))

# remove zero
arr = arr[np.all(arr != 0, axis=1)]
# print(arr)

sliced = arr[:, 0]
# print(sliced)

unique, counts = np.unique(sliced, return_counts=True)
# print(unique.astype('int'), counts)

stacked = np.column_stack((unique.astype('int'), counts))
print(stacked)


table = ('a', 'b', 'c')
for item in stacked:
    print("item: {}  counts:{}".format(table[item[0]], item[1]))
