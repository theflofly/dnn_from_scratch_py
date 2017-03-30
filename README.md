# About

This project provides the python code that support the (soon published) blog post (if you are a beginner, you should read it).

The goal is to make a neural network from scratch using numpy, then the same one using TensorFlow.

As a toy example, we try to predict the price of car using online data.

`download_lbc_cars_data.py` downloads data from leboncoin.fr, which is a website of classified ads. The data retrieved are about BMW Serie 1 (only one model of car).

For each BMW Serie 1 we save an input with the number of km, fuel, age and the price. The data are saved into `car_features.csv`.

These data are then normalized by `normalize_lbc_cars_data.py` to produce `normalized_car_features.csv`.

`normalized_car_features.csv` is used as input by `dnn_from_scratch.py` which is the neural network using numpy and `dnn_from_scratch_tensorflow.py` which is the neural network using TensorFlow.

Overall results are pretty good knowing that the price is impacted by more than three attributes.

# Network architecture
# Usage