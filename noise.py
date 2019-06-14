import h5py
import numpy as np
from PIL import Image
import cv2
import random
f = h5py.File('data.h5', 'r')
images = f['testX'][:, :]
labels = f['testY'][:]
f.close()
N = labels_t.shape[0]
def h(i):
    if i<0:
        return 0
    elif i>255:
        return 255
    else:
        return i

def add_sound(b, k):
    c = []
    for i in range(b.shape[0]):
        d=[]
        for j in range(b.shape[1]):
            hh=h(b[i,j]-random.uniform(-1,1)*k)
            d.append(hh)
        c.append(d)
    c=np.array(c, dtype = 'uint64')
    return c

np_im1 = add_sound(images[25, :].reshape(48, 48, 1), 10)
cv2.imwrite("%d_0.jpg" % 25, np_im1)
np_im2 = add_sound(images[25, :].reshape(48, 48, 1), 20)
cv2.imwrite("%d_1.jpg" % 25, np_im2)
np_im3 = add_sound(images[25, :].reshape(48, 48, 1), 30)
cv2.imwrite("%d_2.jpg" % 25, np_im3)
np_im4 = add_sound(images[25, :].reshape(48, 48, 1), 50)
cv2.imwrite("%d_3.jpg" % 25, np_im4)

np_f10 = add_sound(images[0, :].reshape(48, 48, 1), 10).reshape(1, 2304)
for i in range(1, N):
    np_imt = add_sound(images[i, :].reshape(48, 48, 1), 10).reshape(1, 2304)
    np_f10 = np.concatenate((np_f10, np_imt), axis = 0)
print(np_f10.shape)

np_f20 = add_sound(images[0, :].reshape(48, 48, 1), 20).reshape(1, 2304)
for i in range(1, N):
    np_imt = add_sound(images[i, :].reshape(48, 48, 1), 20).reshape(1, 2304)
    np_f20 = np.concatenate((np_f20, np_imt), axis = 0)
print(np_f20.shape)

np_f30 = add_sound(images[0, :].reshape(48, 48, 1), 30).reshape(1, 2304)
for i in range(1, N):
    np_imt = add_sound(images[i, :].reshape(48, 48, 1), 30).reshape(1, 2304)
    np_f30 = np.concatenate((np_f30, np_imt), axis = 0)
print(np_f30.shape)

np_f50 = add_sound(images[0, :].reshape(48, 48, 1), 50).reshape(1, 2304)
for i in range(1, N):
    np_imt = add_sound(images[i, :].reshape(48, 48, 1), 50).reshape(1, 2304)
    np_f50 = np.concatenate((np_f50, np_imt), axis = 0)
print(np_f50.shape)

f = h5py.File('data_t5.h5', 'w')
f['testX1'] = np_f10
f['testX2'] = np_f20
f['testX3'] = np_f30
f['testX4'] = np_f50
f['testY'] = labels

f.close()