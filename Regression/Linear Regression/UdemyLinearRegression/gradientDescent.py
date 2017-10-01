import numpy as np
import matplotlib.pyplot as plt
from simpleLinearRegression import trainData

N = 10
D = 3

X = np.zeros(((N,D)))
X[:, 0] = 1
X[: 5,1] = 1
X[5:,2] = 1

print('This is X')
print(X)

Y = np.array([0] * 5 + [1] * 5)

w = np.random.randn(D) / np.sqrt(D)
learning_rate = 0.001

print('This is w', w)

print('This is Y')
print(Y)


print("asjkdhfkjsd")
print(trainData(X, Y, w, learning_rate, 1000 ))



# finding the w ny regular vector matrix way
#w = np.linalg.solve(X.T.dot(X), X.T.dot(Y)) #this couldn't be calculated b/c X.T.dot(X) is not invertible
# so need to use gradient descent

costs = []


for t in range(1000):
    Yhat = X.dot(w)
    delta = Yhat - Y
    w = w - learning_rate * X.T.dot(delta)
    mse = delta.dot(delta) / N
    costs.append(mse)


plt.plot(costs)
plt.show()
print('final w', w)

plt.plot(Yhat , label = 'Predictions')
plt.plot(Y, label = 'target')
plt.legend()
plt.show()





