#This class is used to explore the dataset and get a better understanding of it. 
# It prints out the shape of the training and test sets, the unique values of the target variable, 
# and the fraction of positive examples. It also prints out the number of features with missing values, 
# the number of features with few unique values, and the number of constant features.
# Leading to a better understanding of the dataset and potential preprocessing steps that may be needed.

import numpy as np
from helpers import load_csv_data

x_train, x_test, y_train, train_ids, test_ids = load_csv_data("dataset")

print("x_train shape:", x_train.shape)
print("x_test shape: ", x_test.shape)
print("y values:", np.unique(y_train))
print("fraction of positives:", np.mean(y_train == 1))

# Missing values (NaN) per feature
nan_frac = np.mean(np.isnan(x_train), axis=0)
print("\nfeatures with >50% NaN:", np.sum(nan_frac > 0.5))
print("features with 0% NaN:  ", np.sum(nan_frac == 0))
print("NaN fraction, 10 worst:", np.round(np.sort(nan_frac)[-10:], 2))

# How many distinct values each feature has (few = probably categorical)
n_unique = np.array([len(np.unique(col[~np.isnan(col)])) for col in x_train.T])
print("\nfeatures with <=2 unique values: ", np.sum(n_unique <= 2))
print("features with 3-10 unique values: ", np.sum((n_unique > 2) & (n_unique <= 10)))
print("features with >10 unique values:  ", np.sum(n_unique > 10))
print("constant features:                ", np.sum(n_unique <= 1))