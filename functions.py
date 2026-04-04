import pandas as pd
import numpy as np
import os

# Does not work, translations error
def textToArray(file):
    with open(file, 'r') as file:
        content = file.read()
    tweets = content.split(delimiter="\n")
    print(tweets)
    return tweets

#tweets = textToArray('Datasets/tweets.txt')
#emojis = textToArray('Datasets/emojis.txt')

data_files = ['Datasets/ConsumerData.csv', 'Datasets/FONEData.csv', 'Datasets/USAddressData.csv', 'Datasets/ZipData.csv']
ConsumerData, FONEData, USAddressData, ZipData = [pd.read_csv(file) for file in data_files]

print(ConsumerData, FONEData, USAddressData, ZipData)
print(ConsumerData.shape[0], FONEData.shape[0], USAddressData.shape[0], ZipData.shape[0])