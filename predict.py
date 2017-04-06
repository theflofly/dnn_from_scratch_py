"""
The goal of this file is to normalize raw car attributes for the prediction and transform the price using the inverse
of the standardize process.
"""
import numpy as np


class Predict:
    """
    We save the data set metadata.
    """
    def __init__(self, mean_km, std_km, mean_age, std_age, min_price, max_price):
        self.mean_km = mean_km
        self.std_km = std_km
        self.mean_age = mean_age
        self.std_age = std_age
        self.min_price = min_price
        self.max_price = max_price

    """
    This method returns the car's data normalized using the data set metadata.
    """
    def input(self, km, fuel, age):
        km = (km - self.mean_km) / self.std_km
        fuel = -1 if fuel == 'Diesel' else 1
        age = (age - self.mean_age) / self.std_age
        return np.matrix([[
            km, fuel, age
        ]])

    """
    This method returns the price in euro from the output of the network. The inverse of the standardize process for the
    price is applied.
    """
    def output(self, price):
        return price * (self.max_price - self.min_price) + self.min_price