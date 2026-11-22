import pandas as pd 
import numpy as np 
from sklearn import metrics
from sklearn.linear_model import Lasso, LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

dataframe= pd.read_csv("D:\\car_price_prediction\\car_price_details.csv")

# print(dataframe['name'].head())
# print(dataframe['name'].nunique())
# print(dataframe['name'].isnull().sum())

brands = sorted(['Land Rover', 'Mercedes-Benz', 'Aston Martin', 'Rolls-Royce', 'Maruti', 'Hyundai', 'Mahindra', 'Tata', 'Honda', 'Toyota', 'Ford', 'Volkswagen', 'Renault', 'Chevrolet', 'Skoda', 'Nissan', 'Datsun', 'Fiat', 'BMW', 'Audi', 'Jaguar', 'Volvo', 'Jeep', 'Kia', 'MG', 'Isuzu', 'Force', 'Porsche', 'Lexus'], key=len, reverse=True)
# print(brands)

def extract_brand(car_name):

    if pd.isna(car_name):
        return 'Unknown'

    car_name = str(car_name).strip()

    for brand in brands:

        if car_name.lower().startswith(brand.lower()):
            return brand

    return 'Other'

dataframe["brand"]= dataframe["name"].apply(extract_brand)

# print(dataframe[["name", "brand"]].head())
# print(dataframe["brand"].value_counts())

def extract_model(row):
    name= str(row["name"])
    brand= row["brand"]

    if brand in ["Unknown", "Other"]:
        return name

    return name[len(brand):].strip()

dataframe["model_variant"]= dataframe.apply(extract_model, axis=1)
# print(dataframe[["name", "brand", "model_variant"]].head())
# print(encoded_data)

# print(dataframe.shape)
# print(dataframe.head())
# print(dataframe.describe())
# print(dataframe.info())
# print(dataframe.isnull().sum().sum())


# dataframe["fuel"]= (dataframe["fuel"].map({"Petrol":0, "Diesel":1, "CNG":2, "LPG": 3, "Electric": 4}))
# # dataframe["seller_type"]= (dataframe["seller_type"].map({"Individual":0, "Dealer":1, "Trustmark Dealer": 2}))
# dataframe["transmission"]= (dataframe["transmission"].map({"Manual":0, "Automatic":1}))
# dataframe["owner"]= (dataframe["owner"].map({"First Owner":0, "Second Owner":1, "Third Owner": 2, "Fourth & Above Owner": 3, "Test Drive Car": 4}))

# fuel_mode= dataframe["fuel"].mode()[0]
# dataframe["fuel"]= dataframe["fuel"].fillna(fuel_mode)

# # print(dataframe.shape)
# # print(dataframe.head())
# # print(dataframe.describe())
# # print(dataframe.info())
# # print(dataframe.isnull().sum().sum())

drop_column= ["name"]
dataframe= dataframe.drop(columns= drop_column)

# # print(dataframe.head())

target_column= "selling_price"
x= dataframe.drop(columns= [target_column])
y= dataframe[target_column]

# # correlation_matrix = dataframe.corr(method='pearson')
# # print(correlation_matrix)

# plt.figure(figsize=(9, 7))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, square=True, vmin=-1, vmax=1)
# plt.title('Pearson Correlation Heatmap - Car Price Prediction', fontsize=14)
# plt.show()

# # print(x)
# # print(y)
# # print(x.isnull().sum())
# # print(x.isnull().sum().sum())
# # print(x.isnull().value_counts())

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.1, random_state= 2)

categorical_features = ["fuel", "seller_type", "transmission", "owner", "brand", "model_variant"]
numerical_features = ["year", "km_driven"]

encoded_data= OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=False)

x_train_encoded= encoded_data.fit_transform(x_train[categorical_features])
x_test_encoded= encoded_data.transform(x_test[categorical_features])

x_train_final= np.concatenate([x_train[numerical_features].to_numpy(), x_train_encoded], axis=1)
x_test_final= np.concatenate([x_test[numerical_features].to_numpy(), x_test_encoded], axis=1)

# linear_regression_model= LinearRegression()

# linear_regression_model.fit(x_train_final, y_train)
# training_data_prediction= linear_regression_model.predict(x_train_final) 

# print(metrics.r2_score(y_train, training_data_prediction))
# print(metrics.mean_absolute_error(y_test, training_data_prediction))
# print(metrics.mean_squared_error(y_test, training_data_prediction))
# print(metrics.mean_absolute_percentage_error(y_test, training_data_prediction))
# print(metrics.root_mean_squared_error(y_test, training_data_prediction))

# plt.figure(figsize=(8, 6))
# plt.scatter(y_train, training_data_prediction)
# plt.xlabel("Actual Prices")
# plt.ylabel("Predicted Prices")
# plt.title("Actual vs Predicted Prices")
# plt.show()

# testing_data_prediction= linear_regression_model.predict(x_test_final)

# print(metrics.r2_score(y_test, testing_data_prediction))

# plt.figure(figsize=(8, 6))
# plt.scatter(y_test, testing_data_prediction)
# plt.xlabel("Actual Prices")
# plt.ylabel("Predicted Prices")
# plt.title("Actual vs Predicted Prices")
# plt.show()

## ((Linear Regression performs well on positively co-related datasets))


# dataframe["fuel"]= (dataframe["fuel"].map({"Petrol":0, "Diesel":1, "CNG":2, "LPG": 3, "Electric": 4}))
# dataframe["seller_type"]= (dataframe["seller_type"].map({"Individual":0, "Dealer":1, "Trustmark Dealer": 2}))
# dataframe["transmission"]= (dataframe["transmission"].map({"Manual":0, "Automatic":1}))
# dataframe["owner"]= (dataframe["owner"].map({"First Owner":0, "Second Owner":1, "Third Owner": 2, "Fourth & Above Owner": 3, "Test Drive Car": 4}))

# fuel_mode= dataframe["fuel"].mode()[0]
# dataframe["fuel"]= dataframe["fuel"].fillna(fuel_mode)

# drop_column= ["name"]
# dataframe= dataframe.drop(columns= drop_column)

# target_column= "selling_price"
# x= dataframe.drop(columns= [target_column])
# y= dataframe[target_column]

# x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.1, random_state= 2)

lasso_regression_model= Lasso()

lasso_regression_model.fit(x_train_final, y_train)
training_data_prediction= lasso_regression_model.predict(x_train_final) 

print(metrics.r2_score(y_train, training_data_prediction))

plt.figure(figsize=(8, 6))
plt.scatter(y_train, training_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()

testing_data_prediction= lasso_regression_model.predict(x_test_final)

print(metrics.r2_score(y_test, testing_data_prediction))

plt.figure(figsize=(8, 6))
plt.scatter(y_test, testing_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()