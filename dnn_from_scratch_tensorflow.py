import csv
import numpy as np
import tensorflow as tf
import predict as util

# read the data from the CSV
reader = csv.reader(open("normalized_car_features.csv", "r"), delimiter=",")
x = list(reader)
features = np.array(x[2:]).astype("float")
np.random.shuffle(features)

predict = util.Predict(float(x[0][0]), float(x[0][1]), float(x[0][2]), float(x[0][3]), float(x[0][4]),
                                    float(x[0][5]))

data_x = features[:, :3]
data_y = features[:, 3:]

# size of the dataset
m = float(features.shape[0])

# size of the train set
train_set_size = int(m * 0.8)

# the data are splitted between the train and test set
x_data, x_test = data_x[:train_set_size, :], data_x[train_set_size:, :]
y_data, y_test = data_y[:train_set_size, :], data_y[train_set_size:, :]

# regularization strength
Lambda = 0.01
learning_rate = 0.01

with tf.name_scope('input'):
    # training data
    x = tf.placeholder("float", name="cars")
    y = tf.placeholder("float", name="prices")

with tf.name_scope('weights'):
    w1 = tf.Variable(tf.random_normal([3, 3]), name="W1")
    w2 = tf.Variable(tf.random_normal([3, 2]), name="W2")
    w3 = tf.Variable(tf.random_normal([2, 1]), name="W3")

with tf.name_scope('biases'):
    # biases (we separate them from the weights because it is easier to do that when using TensorFlow)
    b1 = tf.Variable(tf.random_normal([1, 3]), name="b1")
    b2 = tf.Variable(tf.random_normal([1, 2]), name="b2")
    b3 = tf.Variable(tf.random_normal([1, 1]), name="b3")

with tf.name_scope('layer_1'):
    # three hidden layer
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, w1), b1))

with tf.name_scope('layer_2'):
    layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, w2), b2))

with tf.name_scope('layer_3'):
    layer_3 = tf.nn.tanh(tf.add(tf.matmul(layer_2, w3),  b3))

with tf.name_scope('regularization'):
    # L2 regularization applied on each weight
    regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3)

with tf.name_scope('loss'):
    # loss function + regularization value
    loss = tf.reduce_mean(tf.square(layer_3 - y)) + Lambda * regularization
    loss = tf.Print(loss, [loss], "loss")

with tf.name_scope('train'):
    # we'll use gradient descent as optimization algorithm
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# launching the previously defined model begins here
init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)

    # we'll make 5000 gradient descent iteration
    for i in range(10000):
        session.run(train_op, feed_dict={x: x_data, y: y_data})

    # testing the network
    print("Testing data")
    print("Loss: " + str(session.run([layer_3, loss], feed_dict={x: x_test, y: y_test})[1]))

    # do a forward pass
    feed_dict = {x: predict.input(168000, "Diesel", 5)}
    print("Predicted price: " + str(predict.output(session.run(layer_3, feed_dict))))

writer = tf.summary.FileWriter('tensorboard', graph=tf.get_default_graph())