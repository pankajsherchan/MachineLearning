import numpy as np
import matplotlib.pyplot as plt

#load data

X = []
Y = []

path = '/Users/Pankaj/machine_learning_examples/linear_regression_class/'

for line in open(path + 'data_1d.csv'):
    x,y = line.split(',')
    X.append(float(x))
    Y.append(float(y))


#let's turn them in numpy array
X = np.array(X)
Y = np.array(Y)

plt.xlabel('X')
plt.ylabel('Y')

plt.scatter(X,Y)

#apply the equation to get m and c
m,c = np.polyfit(X, Y, 1)

print(m,c)
Yhat = 2* X  -10
plt.plot(X, Yhat)
plt.show()

#calculate the R2 or Correlation Coefficient

d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1-  (d1.dot(d1) / d2.dot(d2))

print (r2)
