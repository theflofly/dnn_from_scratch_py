import csv
import tensorflow as tf
import predict as util

reader = csv.reader(open("normalized_car_features.csv", "r"), delimiter=",")
x = list(reader)

predict = util.Predict(float(x[0][0]), float(x[0][1]), float(x[0][2]), float(x[0][3]), float(x[0][4]),
                       float(x[0][5]))

saved_model_directory = "saved_model"

with tf.Session() as session:
    tf.saved_model.loader.load(session, ["dnn_from_scratch_tensorflow"], saved_model_directory)

    # do a forward pass
    print("Predicted price: " + str(predict.output(session.run('layer_3/Tanh:0',
                                                               {'input/cars:0': predict.input(168000, "Diesel", 5)}))))
