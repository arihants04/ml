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

features = (features - features.mean()) / features.std() #this is called normalization, ensures all features contribute to GD

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

# The gradient ssr function
def ssr_gradient(features, actual_value, weights):
    predictions = np.dot(features, weights)
    residuals = predictions - actual_value.values.ravel() #this function ravel turns actual_value into an array
    gradient_w0 = residuals.mean()
    gradient_w = np.dot(features[:,1:].T, residuals) / len(actual_value)
    return np.concatenate(([gradient_w0], gradient_w))

# Implement gradient descent
def gradient_descent(gradient, features, actual_value, start, learn_rate=.001, n_iter=10000, tolerance=1e-6):
    weights = np.array(start)
    for i in range(n_iter):
        grad = gradient(features, actual_value, weights)
        new_weights = weights - learn_rate * grad
        if np.all(np.abs(new_weights - weights) <= tolerance):
            break
        weights = new_weights
    return weights

# Add a column of ones for the intercept terms
features_train_with_intercept = np.c_[np.ones(features_train.shape[0]), features_train]
features_test_with_intercept = np.c_[np.ones(features_test.shape[0]), features_test]

# Initialize weights to 0
initial_weights = np.zeros(features_train_with_intercept.shape[1])

# Gradient descent
optimal_weights = gradient_descent(ssr_gradient, features_train_with_intercept, actual_value_train, initial_weights, learn_rate=0.001, n_iter=10000, tolerance=1e-6)

print("Optimal weights:", optimal_weights)

# Test on test data
predictions = np.dot(features_test_with_intercept, optimal_weights)
# MSE
mse = np.mean((predictions - actual_value_test.values.ravel()) ** 2)
print("MSE:", mse)

# Calculate R² Score
ss_tss = np.sum((actual_value_test.values.ravel() - np.mean(actual_value_test.values.ravel())) ** 2)
ss_rss = np.sum((actual_value_test.values.ravel() - predictions) ** 2)
r2_score = 1 - (ss_rss / ss_tss)
print("R² Score:", r2_score)
