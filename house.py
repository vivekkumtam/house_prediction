import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("housing.csv")
data.dropna(inplace=True)

print(data[['latitude', 'longitude', 'median_house_value']].isnull().sum())
print(data.columns)
print(data[['latitude', 'longitude']].dtypes)
print("Number of rows in the dataset:", len(data))
print(data.info())

x = data.drop(['median_house_value'], axis=1)
y = data['median_house_value']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

train_data = x_train.join(y_train)
train_data = train_data.join(pd.get_dummies(train_data['ocean_proximity'])).drop(['ocean_proximity'], axis=1)

plt.figure(figsize=(15, 8))
sb.scatterplot(x='latitude', y='longitude', data=train_data, hue='median_house_value', palette='coolwarm')
plt.show()

plt.figure(figsize=(15, 8))
sb.heatmap(train_data.corr(), annot=True, cmap="YlGnBu")
plt.show()

train_data['bedroom_ratio'] = train_data['total_bedrooms'] / train_data['total_rooms']
train_data['household_rooms'] = train_data['total_rooms'] / train_data['households']

x_train = train_data.drop(['median_house_value'], axis=1)
y_train = train_data['median_house_value']

reg = LinearRegression()
reg.fit(x_train, y_train)

test_data = x_test.join(y_test)
test_data = test_data.join(pd.get_dummies(test_data['ocean_proximity'])).drop(['ocean_proximity'], axis=1)

test_data['bedroom_ratio'] = test_data['total_bedrooms'] / test_data['total_rooms']
test_data['household_rooms'] = test_data['total_rooms'] / test_data['households']

x_test, y_test = test_data.drop(['median_house_value'], axis=1), test_data['median_house_value']

print("The score for linear regression is:", reg.score(x_test, y_test))

forest = RandomForestRegressor()
forest.fit(x_train, y_train)

print("The score for Random Forest is:", forest.score(x_test, y_test))
