import numpy as np
import csv
import predict as util


class NeuralNetwork:
    def __init__(self):

        # load the dataset from the CSV file
        reader = csv.reader(open("normalized_car_features.csv", "r"), delimiter=",")
        x = list(reader)
        features = np.array(x[2:]).astype("float")
        np.random.shuffle(features)

        # car attribute and price are splitted, note that 1 is appended at each car for the bias
        data_x = np.concatenate((features[:, :3], np.ones((features.shape[0], 1))), axis=1)
        data_y = features[:, 3:]

        # we save the dataset metadata for the prediction part of the network
        self.predict = util.Predict(float(x[0][0]), float(x[0][1]), float(x[0][2]), float(x[0][3]), float(x[0][4]),
                                    float(x[0][5]))

        # we set a threshold at 80% of the data
        self.m = float(features.shape[0])
        self.m_train_set = int(self.m * 0.8)

        # we split the train and test set using the threshold
        self.x, self.x_test = data_x[:self.m_train_set, :], data_x[self.m_train_set:, :]
        self.y, self.y_test = data_y[:self.m_train_set, :], data_y[self.m_train_set:, :]

        # we init the network parameters
        self.z2, self.a2, self.z3, self.a3, self.z4, self.a4 = (None,) * 6
        self.delta2, self.delta3, self.delta4 = (None,) * 3
        self.djdw1, self.djdw2, self.djdw3 = (None,) * 3
        self.gradient, self.numericalGradient = (None,) * 2
        self.Lambda = 0.01
        self.learning_rate = 0.01

        # we init the weights using the blog post values
        self.w1 = np.matrix([
            [0.01, 0.05, 0.07],
            [0.2, 0.041, 0.11],
            [0.04, 0.56, 0.13],
            [0.1, 0.1, 0.1]
        ])

        self.w2 = np.matrix([
            [0.04, 0.78],
            [0.4, 0.45],
            [0.65, 0.23],
            [0.1, 0.1]
        ])

        self.w3 = np.matrix([
            [0.04],
            [0.41],
            [0.1]
        ])

    def forward(self):

        # first layer
        self.z2 = np.dot(self.x, self.w1)
        self.a2 = np.tanh(self.z2)

        # we add the the 1 unit (bias) at the output of the first layer
        ba2 = np.ones((self.x.shape[0], 1))
        self.a2 = np.concatenate((self.a2, ba2), axis=1)

        # second layer
        self.z3 = np.dot(self.a2, self.w2)
        self.a3 = np.tanh(self.z3)

        # we add the the 1 unit (bias) at the output of the second layer
        ba3 = np.ones((self.a3.shape[0], 1))
        self.a3 = np.concatenate((self.a3, ba3), axis=1)

        # output layer, prediction of our network
        self.z4 = np.dot(self.a3, self.w3)
        self.a4 = np.tanh(self.z4)

    def backward(self):

        # gradient of the cost function with regards to W3
        self.delta4 = np.multiply(-(self.y - self.a4), tanh_prime(self.z4))
        self.djdw3 = (self.a3.T * self.delta4) / self.m_train_set + self.Lambda * self.w3

        # gradient of the cost function with regards to W2
        self.delta3 = np.multiply(self.delta4 * self.w3.T, tanh_prime(np.concatenate((self.z3, np.ones((self.z3.shape[0], 1))), axis=1)))
        self.djdw2 = (self.a2.T * np.delete(self.delta3, 2, axis=1)) / self.m_train_set + self.Lambda * self.w2

        # gradient of the cost function with regards to W1
        self.delta2 = np.multiply(np.delete(self.delta3, 2, axis=1) * self.w2.T, tanh_prime(np.concatenate((self.z2, np.ones((self.z2.shape[0], 1))), axis=1)))
        self.djdw1 = (self.x.T * np.delete(self.delta2, 3, axis=1)) / self.m_train_set + self.Lambda * self.w1

    def update_gradient(self):
        self.w1 -= self.learning_rate * self.djdw1
        self.w2 -= self.learning_rate * self.djdw2
        self.w3 -= self.learning_rate * self.djdw3

    def cost_function(self):
        return 0.5 * sum(np.square((self.y - self.a4))) / self.m_train_set + (self.Lambda / 2) * (
            np.sum(np.square(self.w1)) +
            np.sum(np.square(self.w2)) +
            np.sum(np.square(self.w3))
        )

    def set_weights(self, weights):
        self.w1 = np.reshape(weights[0:12], (4, 3))
        self.w2 = np.reshape(weights[12:20], (4, 2))
        self.w3 = np.reshape(weights[20:23], (3, 1))

    def compute_gradients(self):
        nn.forward()
        nn.backward()
        self.gradient = np.concatenate((self.djdw1.ravel(), self.djdw2.ravel(), self.djdw3.ravel()), axis=1).T

    def compute_numerical_gradients(self):
        weights = np.concatenate((self.w1.ravel(), self.w2.ravel(), self.w3.ravel()), axis=1).T

        self.numericalGradient = np.zeros(weights.shape)
        perturbation = np.zeros(weights.shape)
        e = 1e-4

        for p in range(len(weights)):
            # Set perturbation vector
            perturbation[p] = e

            self.set_weights(weights + perturbation)
            self.forward()
            loss2 = self.cost_function()

            self.set_weights(weights - perturbation)
            self.forward()
            loss1 = self.cost_function()

            self.numericalGradient[p] = (loss2 - loss1) / (2 * e)

            perturbation[p] = 0

        self.set_weights(weights)

    def check_gradients(self):
        self.compute_gradients()
        self.compute_numerical_gradients()
        print("Gradient checked: " + str(np.linalg.norm(self.gradient - self.numericalGradient) / np.linalg.norm(
            self.gradient + self.numericalGradient)))

    def predict(self, X):
        self.x = X
        self.forward()
        return self.a4

    def r2(self):
        y_mean = np.mean(self.y)
        ss_res = np.sum(np.square(self.y - self.a4))
        ss_tot = np.sum(np.square(self.y - y_mean))
        return 1 - (ss_res / ss_tot)

    def summary(self, step):
        print("Iteration: %d, Loss %f" % (step, self.cost_function()))
        print("RMSE: " + str(np.sqrt(np.mean(np.square(self.a4 - self.y)))))
        print("MAE: " + str(np.sum(np.absolute(self.a4 - self.y)) / self.m_train_set))
        print("R2: " + str(self.r2()))

    def predict_price(self, km, fuel, age):
        self.x = np.concatenate((self.predict.input(km, fuel, age), np.ones((1, 1))), axis=1)
        nn.forward()
        print("Predicted price: " + str(self.predict.output(self.a4[0])))


def tanh_prime(x):
    return 1.0 - np.square(np.tanh(x))


nn = NeuralNetwork()

print("### Gradient checking ###")
nn.check_gradients()

print("### Training data ###")
nb_it = 5000
for step in range(nb_it):

    nn.forward()
    nn.backward()
    nn.update_gradient()

    if step % 100 == 0:
        nn.summary(step)

print("### Testing data ###")
nn.x = nn.x_test
nn.y = nn.y_test
nn.forward()

print("### Testing summary ###")
nn.summary(nb_it)

print("### Predict ###")
nn.predict_price(168000, "Diesel", 5)
