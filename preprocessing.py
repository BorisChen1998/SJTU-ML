import pandas
import numpy as np
import h5py
filename = './fer2013.csv'

df = pandas.read_csv(filename)
total_size = len(df['emotion'])
train = []
valid = []
test = []
train_y = []
valid_y = []
test_y = []
for i in range(total_size):
    x = df['pixels'][i].split(' ')
    x = [int(item) for item in x]
    y = int(df['emotion'][i])
    if df['Usage'][i] == 'Training':
        train.append(x)
        train_y.append(y)
    elif df['Usage'][i] == 'PublicTest':
        valid.append(x)
        valid_y.append(y)
    elif df['Usage'][i] == 'PrivateTest':
        test.append(x)
        test_y.append(y)

train = np.array(train)
valid = np.array(valid)
test = np.array(test)
train_y = np.array(train_y)
valid_y = np.array(valid_y)
test_y = np.array(test_y)

f = h5py.File('data.h5', 'w')
f['trainX'] = train
f['validX'] = valid
f['testX'] = test
f['trainY'] = train_y
f['validY'] = valid_y
f['testY'] = test_y
f.close()