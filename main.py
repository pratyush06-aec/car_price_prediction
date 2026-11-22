import pandas as pd 
import numpy as np 
from sklearn import metrics
from sklearn.linear_model import Lasso, LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

dataframe= pd.read_csv("D:\\car_price_prediction\\car_price_details.csv")

# print(dataframe.shape)
# print(dataframe.head())
# print(dataframe.describe())
# print(dataframe.info())
# print(dataframe.isnull().sum().sum())


dataframe["fuel"]= (dataframe["fuel"].map({"Petrol":0, "Diesel":1, "CNG":2, "LPG": 3, "Electronic": 4}))
dataframe["seller_type"]= (dataframe["seller_type"].map({"Individual":0, "Dealer":1, "Trustmark Dealer": 2}))
dataframe["transmission"]= (dataframe["transmission"].map({"Manual":0, "Automatic":1}))
dataframe["owner"]= (dataframe["owner"].map({"First Owner":0, "Second Owner":1, "Third Owner": 2, "Fourth & Above Owner": 3, "Test Drive Car": 4}))

fuel_mode= dataframe["fuel"].mode()[0]
dataframe["fuel"]= dataframe["fuel"].fillna(fuel_mode)

# print(dataframe.shape)
# print(dataframe.head())
# print(dataframe.describe())
# print(dataframe.info())
# print(dataframe.isnull().sum().sum())

drop_column= ["name"]
dataframe= dataframe.drop(columns= drop_column)

# print(dataframe.head())

target_column= "selling_price"
x= dataframe.drop(columns= [target_column])
y= dataframe[target_column]

# print(x)
# print(y)
# print(x.isnull().sum())
# print(x.isnull().sum().sum())
# print(x.isnull().value_counts())

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.1, random_state= 2)

linear_regression_model= LinearRegression()

linear_regression_model.fit(x_train, y_train)
training_data_prediction= linear_regression_model.predict(x_test) 

error_score= metrics.r2_score(y_test, training_data_prediction)
print(error_score)