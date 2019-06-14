import h5py
import numpy as np
from PIL import Image
import cv2
import random
f = h5py.File('data.h5', 'r')
images = f['testX'][:, :]
labels = f['testY'][:]
f.close()
N = labels.shape[0]

def flip(arr):
    for m in arr:
        for j in range(48 // 2):
            m[j], m[48-1-j] = m[48-1-j], m[j]
    return arr 

np_f = flip(images[0, :].reshape(48, 48)).reshape(1, 2304)
for i in range(1, N):
    np_imt = flip(images[i, :].reshape(48, 48)).reshape(1, 2304)
    np_f = np.concatenate((np_f, np_imt), axis = 0)

print(np_f.shape)

f = h5py.File('data_t1.h5', 'w')
f['testX'] = np_f
f['testY'] = labels
f.close()
