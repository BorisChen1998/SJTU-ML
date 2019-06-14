import tensorflow as tf
import numpy as np
import random
import h5py
import os
import moxing.tensorflow as mox

data_dir = 'data.h5'
model_dir = 'model/model.ckpt'

batch_size = 128
epochs = 100
lr = 1e-2

def random_batch(l, batch_size):
  rnd_indices = np.random.randint(0, l, batch_size)
  return rnd_indices

def weight_variable(shape):
  initial = tf.truncated_normal(shape,stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1,shape=shape)
  return tf.Variable(initial)

def conv2d(x,W):
  #strides[1,x_movement,y_movement,1]
  return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

f = h5py.File('drive/data.h5', 'r')
trainX = f['trainX'][:, :].astype(np.float32)
trainY_raw_single = f['trainY'][:]
validX = f['validX'][:, :].astype(np.float32)
validY_raw = f['validY'][:]
testX = f['testX'][:, :].astype(np.float32)
testY_raw = f['testY'][:]
f.close()

trainY_raw = np.append(trainY_raw_single, trainY_raw_single)

n_train = trainX.shape[0]
n_valid = validX.shape[0]
n_test = testX.shape[0]
trainX = trainX.reshape(n_train, 48, 48, 1)
trainX_flip = trainX[:, :, ::-1, :]

trainX = np.vstack((trainX, trainX_flip))
validX = validX.reshape(n_valid, 48, 48, 1)
testX = testX.reshape(n_test, 48, 48, 1)
D = trainX.shape[1]
num_class = np.max(trainY_raw) - np.min(trainY_raw) + 1
trainY = np.eye(num_class)[trainY_raw].astype(np.float32)
validY = np.eye(num_class)[validY_raw].astype(np.float32)
testY = np.eye(num_class)[testY_raw].astype(np.float32)

xs = tf.placeholder(tf.float32, [None, 48, 48, 1])
ys = tf.placeholder(tf.float32, [None, num_class])
keep_prob = tf.placeholder(tf.float32) # rate = 1 - keep_prob

#conv1 layer
with tf.name_scope('conv1'):
  w_conv1 = weight_variable([3,3,1,64])
  b_conv1 = bias_variable([64])
  h_conv1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(xs, w_conv1)+b_conv1))
  h_pool1 = max_pool_2x2(h_conv1)
  h_1 = tf.nn.dropout(h_pool1, rate=1-keep_prob)

#conv2 layer
with tf.name_scope('conv2'):
  w_conv2 = weight_variable([5,5,64,128])
  b_conv2 = bias_variable([128])
  h_conv2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_1, w_conv2)+b_conv2))
  h_pool2 = max_pool_2x2(h_conv2)
  h_2 = tf.nn.dropout(h_pool2, rate=1-keep_prob)

#conv3 layer
with tf.name_scope('conv3'):
  w_conv3 = weight_variable([3,3,128,512])
  b_conv3 = bias_variable([512])
  h_conv3 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_2,w_conv3)+b_conv3))
  h_pool3 = max_pool_2x2(h_conv3)
  h_3 = tf.nn.dropout(h_pool3, rate=1-keep_prob)

#conv4 layer
with tf.name_scope('conv4'):
  w_conv4 = weight_variable([3,3,512,512])
  b_conv4 = bias_variable([512])
  h_conv4 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_3,w_conv4)+b_conv4))
  h_pool4 = max_pool_2x2(h_conv4)
  h_4 = tf.nn.dropout(h_pool4, rate=1-keep_prob)

#flatten
with tf.name_scope('flatten'):
  h_flat = tf.reshape(h_4,[-1,3*3*512])

#fc1 layer
with tf.name_scope('fc1'):
  w_fc1 = weight_variable([3*3*512,256])
  b_fc1 = bias_variable([256])
  h_fc1 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(h_flat, w_fc1)+b_fc1))
  h_fc1_drop = tf.nn.dropout(h_fc1, rate=1-keep_prob)

#fc2 layer
with tf.name_scope('fc2'):
  w_fc2 = weight_variable([256,512])
  b_fc2 = bias_variable([512])
  h_fc2 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(h_fc1_drop, w_fc2)+b_fc2, training=True))
  h_fc2_drop = tf.nn.dropout(h_fc2, rate=1-keep_prob)

#output
with tf.name_scope('output'):
  w_output = weight_variable([512,num_class])
  b_output = bias_variable([num_class])
  output = tf.matmul(h_fc2_drop, w_output)+b_output
  sigmoid_output = tf.nn.sigmoid(output)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=ys))
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

'''
saver = tf.train.Saver()
valid_arr, test_arr = [], []
for epoch in range(epochs):
  indices = np.array(random.sample(range(trainX.shape[0]), trainX.shape[0]))
  #train_loss, train_pred = sess.run([loss, output], feed_dict={xs:trainX, ys:trainY, keep_prob:1.0})
  valid_loss, valid_pred = sess.run([loss, output], feed_dict={xs:validX, ys:validY, keep_prob:1.0})
  valid_pred = np.argmax(valid_pred, axis=1)
  valid_acc = np.mean(validY_raw == valid_pred)
  test_pred = sess.run(output, feed_dict={xs:testX, ys:testY, keep_prob:1.0})
  test_pred = np.argmax(test_pred, axis=1)
  test_acc = np.mean(testY_raw == test_pred)
  valid_arr.append(valid_acc)
  test_arr.append(test_acc)
  print("Epoch: ", epoch, " validation loss: ", valid_loss, " validation acc: ", valid_acc, " test acc: ", test_acc)
  
  for iter in range(trainX.shape[0] // batch_size):
    feature_batch, label_batch = trainX[indices[iter*batch_size:(iter+1)*batch_size], :], trainY[indices[iter*batch_size:(iter+1)*batch_size], :]
    _, train_loss = sess.run([train_step, loss], feed_dict={xs:feature_batch, ys:label_batch, keep_prob:0.75})

saver.save(sess, 'drive/ML/model/model.ckpt')
'''

saver = tf.train.Saver()
saver.restore(sess, model_dir)
test_pred = sess.run(output, feed_dict={xs:testX, ys:testY, keep_prob:1.0})
test_pred = np.argmax(test_pred, axis=1)
test_acc = np.mean(testY_raw == test_pred)
print("Test acc: ", test_acc)