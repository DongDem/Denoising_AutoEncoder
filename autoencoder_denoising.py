import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#matplotlib inline
import tensorflow.contrib.layers as layers
import tensorflow.contrib.framework as ops
import os
import cv2
import time

from tensorflow.examples.tutorials.mnist import input_data
slim = tf.contrib.slim
start_time = time.time()

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
        image = np.array(image,dtype=np.uint8)
        #image = np.array(image, dtype=np.float32)
        image = image.astype('float32')
        image = image / 255
        image_flatten = image.flatten()
        data.append(image_flatten)

    return data


def load_train1(train_path):
    data = []
    for image_file in sorted(os.listdir(train_path)):
        full_dir = os.path.join(train_path, image_file)
        print(full_dir)
        image = tf.read_file(full_dir)
        image = tf.image.decode_jpeg(image, channels=1)
        image =tf.image.resize_images(image,[image_size, image_size])
        image_flatten = tf.layers.flatten(image)
        image_flatten = image_flatten *1.0/127.5- 1.0
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

p= 0.05
def kl_divergence(rho, rho_hat):
    return rho * tf.log(rho) - rho * tf.log(rho_hat) + (1 - rho) * tf.log(1 - rho) - (1 - rho) * tf.log(1 - rho_hat)

def gaussian_additive_noise(x, std):
    return x + tf.random_normal(shape=tf.shape(x), dtype=tf.float32, mean=0.0, stddev=std)

imgs = tf.placeholder(tf.float32, shape=[None, 4096], name="Input")
lr = tf.placeholder(tf.float32)
noise_images = gaussian_additive_noise(imgs, 10/255)
'''
# local layer weight initialization
def local_weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.04)
    return tf.Variable(initial)

def local_bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

W_fc2 = local_weight_variable([2048, 4096])
b_fc2 = local_bias_variable([4096])
decoder = tf.matmul(encoder1, W_fc2) + b_fc2
'''

#encoder1 = layers.fully_connected(noise_images, 2048, activation_fn=tf.nn.relu,weights_initializer=layers.xavier_initializer())
encoder1 = tf.layers.dense(noise_images, 2048, activation=tf.nn.sigmoid)

p_hat1=tf.reduce_mean(encoder1,axis=0)
kl_div1 = kl_divergence(p, p_hat1)

#encoder2 = layers.fully_connected(encoder1, 2048, activation_fn=tf.nn.relu,weights_initializer=layers.xavier_initializer(),scope='encoder2')
encoder2 = tf.layers.dense(encoder1, 1024, activation=tf.nn.sigmoid)
p_hat2=tf.reduce_mean(encoder2,axis=0)
kl_div2 = kl_divergence(p, p_hat2)
encoder3 = tf.layers.dense(encoder2, 2048, activation=tf.nn.sigmoid)

decoder =tf.layers.dense(encoder3, 4096, activation=None)

loss = tf.reduce_mean(tf.square(decoder - imgs)) + 0.001 * tf.reduce_sum(kl_div1)+ 0.001 * tf.reduce_sum(kl_div2)

# chuan cua learning rate la : 0.001
learning_rate = 0.001
batch_size = 64
n_epochs = 1000
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
n_batchs = trainset._num_examples // batch_size
#n_batchs = mnist.train.num_examples // batch_size
#mnist.train.num_examples
print(n_batchs)

optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
#mean_data = np.mean(trainset.data, axis=0)
#config = tf.ConfigProto(device_count={'GPU': 0})
#sess = tf.Session(config=config)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
def initial_learning_rate(epoch):
    if (epoch >= 0) and (epoch < 800):
        return 0.001
    if (epoch >= 800) and (epoch < 1010):
        return 0.0001
for i_epoch in range(1, n_epochs+1):
    loss_avg = 0.0
    starter_learning_rate = initial_learning_rate(i_epoch)
    for i_batch in range(1, n_batchs+1):
        #batch_xs, _ = mnist.train.next_batch(batch_size)
        b = trainset.next(batch_size)#-mean_data
        _, loss_val = sess.run([optimizer, loss], feed_dict={imgs: b,lr:starter_learning_rate})
       # printed_img = sess.run(encoder1,feed_dict={imgs:b[1].reshape(1,4096)})
       # printed_decoder = sess.run(decoder, feed_dict={imgs: b[1].reshape(1,4096)})
        #print('dongdem')
       # print(printed_img)
       # print(printed_decoder)
        loss_avg = (loss_val / batch_size)
    print(i_epoch, loss_avg)

print(time.time() - start_time)
'''
n_examples = 5
test_xs, _ = mnist.test.next_batch(n_examples)
'''
any_images = testset.data[:5]

print_noise_image, reconstructed = sess.run([noise_images, decoder], feed_dict={imgs: any_images})
reconstructed1 = reconstructed
f1 =plt.figure(1)
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(any_images[i].reshape(64, 64), cmap='gray')
    plt.axis('off')

f2=plt.figure(2)
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(print_noise_image[i].reshape(64, 64), cmap='gray')
    plt.axis('off')

f3=plt.figure(3)
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(reconstructed[i].reshape(64, 64), cmap='gray')
    plt.axis('off')

f4=plt.figure(4)
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(reconstructed1[i].reshape(64, 64), cmap='gray')
    plt.axis('off')
plt.show()
print(time.time() - start_time)