import numpy as np
import matplotlib.pyplot as plt

#load data

X = []
Y = []

path = '/Users/Pankaj/machine_learning_examples/linear_regression_class/'

for line in open(path + 'data_2d.csv'):
    x1,x2,y = line.split(',')
    X.append([float(1), float(x1), float(x2)])
    Y.append(float(y))

#turn X and Y to numpy array
X = np.array(X)
Y = np.array(Y)

#plot the data to see how it looks
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X[:, 0], X[:, 1], Y)
# plt.show()

#calculate the weights
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat = np.dot(X, w)

#calculate the r2
d1 = Y - Yhat
d2 = Y - Y.mean()

r2 = 1 - d1.dot(d1) / d2.dot(d2)
print(r2)


