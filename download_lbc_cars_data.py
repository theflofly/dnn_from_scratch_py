# coding=utf-8

"""
This script download data from leboncoin.fr and save them into a CSV file named car_features.csv.
"""
from bs4 import BeautifulSoup
import requests
import urlparse
import csv

# the BMW1 serie 1 page is used
url = "https://www.leboncoin.fr/voitures/offres/rhone_alpes/occasions/?o=0&brd=Bmw&mdl=Serie%201"
r = requests.get(url)
data = r.text

soup = BeautifulSoup(data, "html.parser")
carLinks = set()
pageLinks = set()
data_set = []

parsed = urlparse.urlparse(soup.select('a#last')[0].get('href'))
nbPage = urlparse.parse_qs(parsed.query)['o'][0]
print("There are " + str(nbPage) + " web pages to process")

# for each web page that contains a grid of car offers
for i in range(1, int(nbPage), 1):

    print("Processing web page: " + str(i))

    # each car offer link is saved into the carLinks
    for link in soup.select('#listingAds > section > section > ul > li > a'):
        carLinks.add(link.get('href').replace("//", "http://"))

    # the next url page is set
    url = "https://www.leboncoin.fr/voitures/offres/rhone_alpes/occasions/?o=" + str(i) + "&brd=Bmw&mdl=Serie%201"
    r = requests.get(url)
    data = r.text
    soup = BeautifulSoup(data, "html.parser")

# for each car link
for carLink in carLinks:

    print("Processing car page: " + carLink)

    # we load the car page
    r = requests.get(carLink)
    data = r.text
    soup = BeautifulSoup(data, "html.parser")
    km = 0
    fuel = ""
    age = 0
    price = 0

    # for each attribute of the car
    for info in soup.select("div.line h2"):

        # we keep the ones that we need
        if info.select('.property')[0].text == u'Kilométrage':
            km = int(info.select('.value')[0].text.replace(" ", "").replace("KM", ""))
        if info.select('.property')[0].text == u'Carburant':
            fuel = info.select('.value')[0].text
        if info.select('.property')[0].text == u'Année-modèle':
            age = 2017 - int(info.select('.value')[0].text)
        if info.select('.property')[0].text == u'Prix':
            price = int(info.select('.value')[0].text.replace(" ", "").replace(u"€", ""))

    # each car is an array of four features added to the data_set
    data_set.append([km, fuel, age, price])

# the data_set is save into the CSV file
fl = open('car_features.csv', 'w')
writer = csv.writer(fl)
writer.writerow(['km', 'fuel', 'age', 'price'])
for values in data_set:
    writer.writerow(values)

fl.close()
