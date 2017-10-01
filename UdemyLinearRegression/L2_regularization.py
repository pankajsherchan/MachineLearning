import numpy as np
import matplotlib.pyplot as plt

N = 50

X = np.linspace(0,10,N)
Y = 0.5 * X + np.random.randn(N)

Y[-1] += 30
Y[-2] += 30

plt.scatter(X, Y)
plt.show()

X = np.vstack([np.ones(N), X]).T

w_ml = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
Yhat_ml = X.dot(w_ml)

plt.scatter(X[:, 1], Y)
plt.plot(X[:, 1], Yhat_ml)

plt.show()

l2 = 1000
w_regularized = np.linalg.solve(l2 * np.eye(2) + X.T.dot(X), X.T.dot(Y))

print('The identity', l2 * np.eye(2))
print('the x transpose dot with x', X.T.dot(X))

Yhat_regularized = X.dot(w_regularized)
plt.scatter(X[:, 1], Y)
plt.plot(X[:, 1], Yhat_ml, label='maximum Likelihood')
plt.plot(X[:, 1], Yhat_regularized, label='regularized')
plt.legend()
plt.show()
