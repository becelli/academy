import numpy as np
# import scipy.signal as ss

arr1 = np.random.random(10_000_000)
arr2 = np.random.random(10_000_000)

print('A (shape): ', arr1.shape)
print('B (shape): ', arr2.shape)
conv = np.convolve(arr1, arr2)
print('C (shape): ', conv.shape)

# conv2 = ss.fftconvolve(a, b)
# print('D (chape): ', conv2.shape)
