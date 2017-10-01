import numpy as np
import matplotlib.pyplot as plt

#load the data
X = []
Y = []

path = '/Users/Pankaj/machine_learning_examples/linear_regression_class/'

for line in open(path + 'data_poly.csv'):
    x,y = line.split(',')
    x = float(x)
    X.append([1, x, x * x])
    Y.append(float(y))

#convert to numpy array
X = np.array(X)
Y = np.array(Y)

#plot the data
plt.scatter(X[:,1], Y)
#plt.show()

#calculate the weight
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))

Yhat = np.dot(X, w)

#plot the data again
#since a quadratic function is monotonic , the sorted data should work fine, infact better for visualization
plt.plot(sorted(X[:, 1]), sorted(Yhat))
plt.show()

#calculate the r2
d1 = Y - Yhat
d2 = Y - Y.mean()

r2 = 1 - d1.dot(d1) / d2.dot(d2)
print(r2)
