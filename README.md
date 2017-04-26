# Note

The project works with python 2.7.

# About

This project provides the python code that supports this [blog post](https://matrices.io/deep-neural-network-from-scratch/) (if you are a beginner, you should read it).

The goal is to make a neural network from scratch using numpy, then the same one using TensorFlow.

As a toy example, we try to predict the price of car using online data.

`download_lbc_cars_data.py` downloads data from leboncoin.fr, which is a website of classified ads. The data retrieved are about BMW Serie 1 (only one model of car).

For each BMW Serie 1 we save an input with the number of km, fuel, age and the price. The data are saved into `car_features.csv`.

These data are then normalized by `normalize_lbc_cars_data.py` to produce `normalized_car_features.csv`.

`normalized_car_features.csv` is used as input by `dnn_from_scratch.py` which is the neural network using numpy and `dnn_from_scratch_tensorflow.py` which is the neural network using TensorFlow.

`predict.py` is used to transform the data back and forth from the normalized to the human readeable version. For instance to predict a price, the user will input the raw car attributes. `predict.py` will convert the raw data to the normalized version and return them. The neural network output is also given to `predict.py` so that the user obtains a readable price and not a normalized one.

Overall results are pretty good knowing that the price is impacted by more than three attributes.

# Network architecture
The architecture is pretty simple and well described in the blog post. Here is an illustration:
![Network architecture](https://matrices.io/content/images/2017/02/DNN-S12.png)

# Usage
A requirements.txt file exists at the root of the repository. Run `pip install -r requirements.txt `.

# Issue
If you see a bad implementation or you come across a bug, open an issue. I'll help you.
