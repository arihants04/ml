import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
  
# fetch dataset 
real_estate_valuation = fetch_ucirepo(id=477) 
  
# data (as pandas dataframes) 
features = real_estate_valuation.data.features 
actual_value = real_estate_valuation.data.targets

# check for duplicate rows and remove them
print("Checking for duplicate rows in features")
print(features.duplicated().sum())

features = features.drop_duplicates()
actual_value = actual_value.loc[features.index]

print(features.head())
print(actual_value.head())

# here I am checking if both tables have the same number of entries
print(features.shape)
print(actual_value.shape)

# this is called normalization, ensures all features contribute to GD
features = (features - features.mean()) / features.std()

# split the dataset
features_train, features_test, actual_value_train, actual_value_test = train_test_split(
    features, actual_value, test_size=0.2, random_state=21)

print("Shape of features_train:", features_train.shape)
print("Shape of features_test:", features_test.shape)
print("Shape of actual_value_train:", actual_value_train.shape)
print("Shape of actual_value_test:", actual_value_test.shape)

# create and train the SGDRegressor
sgd_regressor = SGDRegressor(max_iter=10000, tol=1e-6, learning_rate='constant', eta0=0.007, random_state=21)
sgd_regressor.fit(features_train, actual_value_train.values.ravel())

# optimal weights
optimal_weights = np.concatenate(([sgd_regressor.intercept_[0]], sgd_regressor.coef_))
print("Optimal weights:", optimal_weights)

# test on test data
predictions = sgd_regressor.predict(features_test)

# calculate and print MSE
mse = mean_squared_error(actual_value_test, predictions)
print("MSE:", mse)

# calculate and display R² Score
r2 = r2_score(actual_value_test, predictions)
print("R² Score:", r2)
