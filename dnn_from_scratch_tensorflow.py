import csv
import numpy as np
import tensorflow as tf

reader = csv.reader(open("normalized_car_features.csv", "rb"), delimiter=",")
x = list(reader)
features = np.array(x[1:]).astype("float")
np.random.shuffle(features)

data_x = features[:, :3]
data_y = features[:, 3:]

m = float(features.shape[0])
threshold = int(m * 0.8)

x_data, x_test = data_x[:threshold, :], data_x[threshold:, :]
y_data, y_test = data_y[:threshold, :], data_y[threshold:, :]

x = tf.placeholder("float")
y = tf.placeholder("float")

w1 = np.matrix([
    [0.01, 0.05, 0.07],
    [0.2, 0.041, 0.11],
    [0.04, 0.56, 0.13]
])

w2 = np.matrix([
    [0.04, 0.78],
    [0.4, 0.45],
    [0.65, 0.23]
])

w3 = np.matrix([
    [0.04],
    [0.41]
])

w1 = tf.Variable(w1, dtype=tf.float32)
w2 = tf.Variable(w2, dtype=tf.float32)
w3 = tf.Variable(w3, dtype=tf.float32)

b1 = tf.Variable(np.matrix([0.1, 0.1, 0.1]), dtype=tf.float32)
b2 = tf.Variable(np.matrix([0.1, 0.1]), dtype=tf.float32)
b3 = tf.Variable(np.matrix([0.1]), dtype=tf.float32)

layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, w1), b1))
layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, w2), b2))
layer_3 = tf.nn.tanh(tf.add(tf.matmul(layer_2, w3),  b3))

loss = tf.reduce_sum(tf.square(layer_3 - y))
loss = tf.Print(loss, [loss], "loss")

train_op = tf.train.GradientDescentOptimizer(1/m * 0.01).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    for i in range(10000):
        session.run(train_op, feed_dict={x: x_data, y: y_data})

# TODO: use L2 regularization
# TODO: tensorboard