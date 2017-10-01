import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def trainData(X, Y, theta, learning_rate, iterations):

    #theta = np.linalg.solve( np.dot(X.T, X), np.dot(X.T, Y) )
    print('Initial X lenght', len(X))
    costs = []
    for i in range(iterations):
        Ytheta = X.dot(theta)
        cost = Ytheta - Y
        theta = theta - learning_rate * X.T.dot(cost)
        mse = cost.T.dot(cost) / len(X)
        costs.append(mse)

    plt.plot(costs)
    plt.show()
    return theta


def main():

    print('reading CSV')
    X = []
    Y = []
    path = '/Users/Pankaj/machine_learning_examples/linear_regression_class/'

    for line in open(path + 'data_1d.csv'):
        # print(line)
        x, y = line.split(',')
        #print(x)
        #print(y)
        X.append(float(x))
        Y.append(float(y))

    X = np.array(X)
    Y = np.array(Y)

    plt.scatter(X, Y)
    plt.show()

    N = len(X)
    X = np.vstack([np.ones(N), X]).T

    # m,c = np.polyfit(X, Y, 1)
    # print(m)
    # print(c)

    initial_theta = np.array([0.5, 0.5])

    learning_rate = 0.01
    iterations = 200

    m= trainData(X, Y, initial_theta, learning_rate, iterations)
    print(m)


if __name__ == "__main__":
    main()


