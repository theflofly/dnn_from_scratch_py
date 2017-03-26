import csv
import numpy as np
import tensorflow as tf

reader = csv.reader(open("normalized_car_features.csv", "rb"), delimiter=",")
x = list(reader)
features = np.array(x[1:]).astype("float")
#np.random.shuffle(features)

data_x = features[:, :3]
data_y = features[:, 3:]

m = float(features.shape[0])
threshold = int(m * 0.8)

x, x_test = data_x[:threshold, :], data_x[threshold:, :]
y, y_test = data_y[:threshold, :], data_y[threshold:, :]

x1 = tf.placeholder("float")
y1 = tf.placeholder("float")

w1_ = np.matrix([
    [0.01, 0.05, 0.07],
    [0.2, 0.041, 0.11],
    [0.04, 0.56, 0.13]
])

w2_ = np.matrix([
    [0.04, 0.78],
    [0.4, 0.45],
    [0.65, 0.23]
])

w3_ = np.matrix([
    [0.04],
    [0.41]
])

w1 = tf.Variable(w1_, dtype=tf.float32)
w2 = tf.Variable(w2_, dtype=tf.float32)
w3 = tf.Variable(w3_, dtype=tf.float32)

b1 = tf.Variable(np.matrix([0.1, 0.1, 0.1]), dtype=tf.float32)
b2 = tf.Variable(np.matrix([0.1, 0.1]), dtype=tf.float32)
b3 = tf.Variable(np.matrix([0.1]), dtype=tf.float32)

layer_1 = tf.nn.tanh(tf.add(tf.matmul(x1, w1), b1))
layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, w2), b2))
layer_3 = tf.nn.tanh(tf.add(tf.matmul(layer_2, w3),  b3))

loss = tf.reduce_sum(tf.square(layer_3 - y1))
loss = tf.Print(loss, [loss], "loss")

train_op = tf.train.GradientDescentOptimizer(0.01)
loss_ = train_op.minimize(loss)

init = tf.global_variables_initializer()

var_grad = tf.gradients(loss, [layer_3])[0]

# why the gradient is so high ?

with tf.Session() as session:
    session.run(init)

    for i in range(100):
         session.run(loss_, feed_dict={x1: x, y1: y})

    print(session.run(y1, feed_dict={x1: x, y1: y}))
    print(session.run(layer_3, feed_dict={x1: x, y1: y}))

#train_writer = tf.summary.FileWriter('tensorboard', session.graph)