from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist", one_hot=True)
train_path = './dataset/train_augment/'
test_path = './dataset/test/'
image_size=64
def load_train(train_path):
    data = []
    for image_file in sorted(os.listdir(train_path)):
        full_dir = os.path.join(train_path, image_file)
        print(full_dir)
        image = cv2.imread(full_dir)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
        image = np.array(image)
        image_flatten = image.flatten()
        data.append(image_flatten)
    return data

class DatasetSequence(object):
    def __init__(self, data):
        self.data = data
        self.batch_id = 0
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = len(self.data)

    def next(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finish epoch
            self._index_in_epoch += 1
            # shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self.data = [self.data[e] for e in perm]
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self.data[start:end]

data_train = load_train(train_path)
data_test = load_train(test_path)
trainset = DatasetSequence(data_train)
print(trainset._num_examples)
testset = DatasetSequence(data_test)



# Network Parameters
num_hidden_1 = 2048 # 1st layer num features
num_input = 4096 # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder(tf.float32, shape=[None, 4096], name="Input")

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b1': tf.Variable(tf.random_normal([num_input])),
}

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    return layer_1
# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    return layer_1

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))

lr = 0.0001
batch_size = 64
n_epochs = 500
n_batchs = trainset._num_examples // batch_size
print(n_batchs)
mean_data = np.mean(trainset.data, axis=0)
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

#config = tf.ConfigProto(device_count={'GPU': 0})
#sess = tf.Session(config=config)
sess = tf.Session()
sess.run(tf.global_variables_initializer())


for i_epoch in range(1, n_epochs+1):
    loss_avg = 0.0
    for i_batch in range(1, n_batchs+1):
        b = trainset.next(batch_size)
        _, loss_val = sess.run([optimizer, loss], feed_dict={X: b})
        loss_avg = (loss_val / batch_size)
    print(i_epoch, loss_avg)


any_images = testset.data[:5]
reconstructed = sess.run([decoder_op], feed_dict={X: any_images})
reconstructed = np.array(reconstructed)
reconstructed = reconstructed.reshape([5,4096])
print(reconstructed)
f1 =plt.figure(1)
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(any_images[i].reshape(64, 64), cmap='gray')
    plt.axis('off')
'''
f2=plt.figure(2)
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(print_noise_image[i].reshape(64, 64), cmap='gray')
    plt.axis('off')
'''
f3=plt.figure(3)
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(reconstructed[i].reshape(64, 64), cmap='gray')
    plt.axis('off')
plt.show()
