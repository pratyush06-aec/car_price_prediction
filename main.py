import pandas as pd 
import numpy as np 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
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


# dataframe['']
