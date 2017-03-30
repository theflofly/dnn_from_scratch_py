"""
This script normalize the car dataset and produce the normalized_car_features.csv.
"""
import csv
import numpy as np

reader = csv.reader(open("car_features.csv", "rb"), delimiter=",")
x = list(reader)
features = np.array(x).astype("string")

feature_cleaned = []

# cleaning: we keep the car Diesel and Essence for which the price is higher than 1000 euros
# also removing the headers column
for feature in features[1:, :]:
    if (feature[1] == 'Diesel' or feature[1] == 'Essence') and int(feature[3]) > 1000:
        feature_cleaned.append(feature)

print("Original dataset size: " + str(features.shape[0] - 1))
features = np.array(feature_cleaned).astype("string")
print("Cleaned dataset size: " + str(features.shape[0]))

# standardize kilometers: (x - mean)/std
km = features[:, 0].astype("int")
mean_km = np.mean(km)
std_km = np.std(km)
km = (km - mean_km)/std_km
features[:, 0] = km

# binary convert fuel: Diesel = -1, Essence = 1
features[:, 1] = [-1 if x == 'Diesel' else 1 for x in features[:,1]]

# standardize age: (x - mean)/std
age = features[:, 2].astype("int")
mean_age = np.mean(age)
std_age = np.std(age)
age = (age - mean_age)/std_age
features[:, 2] = age

# standardize price: (x - min)/(max - min)
price = features[:, 3].astype("float")
min_price = np.min(price)
max_price = np.max(price)
features[:, 3] = (price - min_price)/(max_price - min_price)

# summary
print("Mean km: " + str(mean_km))
print("Std km: " + str(std_km))
print("Mean age: " + str(mean_age))
print("Std age: " + str(std_age))
print("Min price: " + str(min_price))
print("Max price: " + str(max_price))

fl = open('normalized_car_features.csv', 'w')

writer = csv.writer(fl)
# the first line contains the normalization metadata
writer.writerow([mean_km, std_km, mean_age, std_age, min_price, max_price])
writer.writerow(['km', 'fuel', 'age', 'price'])
for values in features:
    writer.writerow(values)

fl.close()