import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

HOUSING_PATH = "datasets/housing"


def load_houseing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)

    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


housing = load_houseing_data()
print(housing.head())
print(housing.info())

# histograms
housing.hist(bins=50, figsize=(20, 15))
plt.show()

train_set, test_set = split_train_test(housing, 0.2)
