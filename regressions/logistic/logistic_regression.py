import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


data = pd.read_csv('iris-data.csv')
data.head()
data.describe()
data.info()


data = data.dropna()
data.info()

data['class'].value_counts()
data['class'].replace(["Iris-setossa", "versicolor"],
                      ["Iris-setosa", "Iris-versicolor"], inplace=True)
data['class'].value_counts()


final_data = data[data['class'] != 'Iris-virginica']
final_data['class'].value_counts()

sns.pairplot(final_data, hue='class', height=2.5)


final_data.hist(column='sepal_length_cm', bins=20, figsize=(10, 5))
final_data.loc[final_data.sepal_length_cm < 1, [
    'sepal_length_cm']] = final_data['sepal_length_cm']*100
final_data.hist(column='sepal_length_cm', bins=20, figsize=(10, 5))
final_data = final_data.drop(final_data[(
    final_data['class'] == "Iris-setosa") & (final_data['sepal_width_cm'] < 2.5)].index)


sns.pairplot(final_data, hue='class', size=2.5)


final_data['class'].replace(
    ["Iris-setosa", "Iris-versicolor"], [1, 0], inplace=True)
final_data.head()
