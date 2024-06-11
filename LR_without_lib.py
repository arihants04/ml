import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
real_estate_valuation = fetch_ucirepo(id=477) 
  
# data (as pandas dataframes) 
features = real_estate_valuation.data.features 
actual_value = real_estate_valuation.data.targets 
  
# # metadata 
# print(real_estate_valuation.metadata) 
  
# # variable information 
# print(real_estate_valuation.variables) 

# print("Checking for null values")
# print(features.isnull().sum())
# print(actual_value.isnull().sum())

print("Checking for duplicate rows in features")
print(features.duplicated().sum())

features = features.drop_duplicates()
actual_value = actual_value.loc[features.index]


print(features.head())
print(actual_value.head())

# Here I am checking if both tables have the same number of entries
print(features.shape)
print(actual_value.shape)


def train_test_split(features, actual_value, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)
    indexes = np.random.permutation(len(features))
    print("indexes " + str(indexes))
    test_set_size = int(len(features) * test_size)
    print("test_set_size " + str(test_set_size))
    test_indexes = indexes[:test_set_size]
    print("test_indexes " + str(test_indexes))
    train_indexes = indexes[test_set_size:]
    print("train_indexes " + str(train_indexes))
    return features.iloc[train_indexes], features.iloc[test_indexes], actual_value.iloc[train_indexes], actual_value.iloc[test_indexes]

#split the dataset
features_train, features_test, actual_value_train, actual_value_test = train_test_split(features, actual_value, test_size=0.2, random_state=21)

# checking entries of split datasets
print("Shape of features_train:", features_train.shape)
print("Shape of features_test:", features_test.shape)
print("Shape of actual_value_train:", actual_value_train.shape)
print("Shape of actual_value_test:", actual_value_test.shape)

