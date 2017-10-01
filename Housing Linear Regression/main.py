import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_boston
from sklearn import model_selection
from sklearn.linear_model import LinearRegression



housing_data = load_boston()

# extracting the informaiton out of dataset

print('Extracting the information out of dataset')
print(housing_data.keys())
print(housing_data.data.shape)
print(housing_data.feature_names)
print(housing_data.DESCR)


#converting the housing dataset into pandas dataframe
housing_dataFrame = pd.DataFrame(housing_data.data)


#reading the first five data
print('first 5 housing data')
print(housing_dataFrame.head())

#changing to meaningful column name
housing_dataFrame.columns = housing_data.feature_names

#print(housing_data_dataFrame.head())

#extracting the target variable that contains the actual price
#print(housing_data.target)

print('Extracting the target varibale to price column')
housing_dataFrame['price'] = housing_data.target

#the housingdata dataframe now should have price columns
print(housing_dataFrame.head())

#statistics of the data
print('Printing the statistics of the data')
print(housing_dataFrame.describe())


X = housing_dataFrame.drop('price', axis = 1)
Y = housing_dataFrame['price']

#dividing the dataset into test and train
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.33, random_state = 5)

#shape of train and test data
print('Checking the shape of training and testing data')
print(X_train.shape, X_test.shape)
print(Y_train.shape, Y_test.shape)


#applying linear regression
lr = LinearRegression()
lr.fit(X_train, Y_train)

Y_predict = lr.predict(X_test)

plt.scatter(Y_test, Y_predict)
plt.xlabel("Prices")
plt.ylabel('Predicted Price')
plt.title("Price vs Predicted Price")
plt.show()

#since the data didn't fit perfectly, the graph is not perfectly linear. So calculating the mse(mean squared error for furthur verification)

mse = sklearn.metrics.mean_squared_error(Y_test, Y_predict)
print('Mean squared error of actual y and predicted Y')
print(mse) #28.5413672756 is not good enough. Conclusion: the dataset doesn't possess a good linear combo.


#calculate the R2 or Correlation Coefficient
d1 = Y_test - Y_predict
d2 = Y_test - Y.mean()
r2 = 1-  (d1.dot(d1) / d2.dot(d2))
print('Correlation Coefficient of Actual Y and Predicted Y')
print(r2)
# r2 = 0.695539055169, not a good linear model


