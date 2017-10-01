import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#load data
path = '/Users/Pankaj/machine_learning_examples/linear_regression_class/'
df = pd.read_excel(path + 'mlr02.xls')
X = df.as_matrix()

# The data (x1,x2,x3) for each patient
#x1 = Systolic Pressure
#x2 = age in Years
#x3 = weight in pounds

plt.scatter(X[:, 1], X[:, 0])
plt.show()

plt.scatter(X[:, 2], X[:, 0])
plt.show()

df['ones'] = 1
Y = df['X1']
X = df[['X2', 'X3', 'ones']]

X2Only = df[['X2', 'ones']]
X3Only = df[['X3', 'ones']]


def get_r2(X, Y):
    w = np.linalg.solve( X.T.dot(X), X.T.dot(Y) )
    Yhat = X.dot(w)

    d1 = Y - Yhat
    d2 = Y - Y.mean()
    r2 = 1-  (d1.dot(d1) / d2.dot(d2))
    return r2

print("r2 for X2 only", get_r2(X2Only, Y))
print("r2 for X3 only", get_r2(X3Only, Y))
print("for both", get_r2(X, Y))





#
#

#
# for line in open(path + 'mlr02.xls'):
#     x,y = line.split(',')
#     X.append(float(x))
#     Y.append(float(y))
#
#
# #let's turn them in numpy array
# X = np.array(X)
# Y = np.array(Y)
#
# plt.scatter(X,Y)
#
# #apply the equation to get m and c
# m,c = np.polyfit(X, Y, 1)
#
#
# Yhat = m * X + c
# plt.plot(X, Yhat)
# plt.show()
#
# #calculate the R2 or Correlation Coefficient
#
# d1 = Y - Yhat
# d2 = Y - Y.mean()
# r2 = 1-  (d1.dot(d1) / d2.dot(d2))
#
# print (r2)
