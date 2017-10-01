import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def trainData(X, Y, theta, learning_rate, iterations):

    #theta = np.linalg.solve( np.dot(X.T, X), np.dot(X.T, Y) )

    costs = []
    for i in range(iterations):
        Ytheta = X.dot(theta)
        cost = Ytheta - Y
        theta = theta - learning_rate * X.T.dot(cost)
        mse = cost.dot(cost) / len(X)
        print(mse, 'iteration' , i)
        costs.append(mse)

    # for t in range(1000):
    #     Yhat = X.dot(w)
    #     delta = Yhat - Y
    #     w = w - learning_rate * X.T.dot(delta)
    #     mse = delta.dot(delta) / N
    #     costs.append(mse)

    plt.plot(costs)
    plt.show()
    return theta

def testData(X, Y, theta, originalX):

    Ytheta = X.dot(theta)

    plt.scatter(originalX, Y)
    plt.plot(originalX, Ytheta)
    plt.show()
    return 0

def scale_dataset(X):
    # scale input dataset and return z where z = (x - mean)/standard deviation

    for col in range(1, X.shape[1]):
        mean = np.mean(X[:, col])
        std = np.std(X[:, col])
        X[:, col] = (X[:, col] - mean)/std
    return X

def scale_dataset_y(Y):
        mean = np.mean(Y)
        std = np.std(Y)
        Y = (Y - mean) / std
        return Y



def main():
    print('reading CSV')
    X = []
    Y = []
    path = '/Users/Pankaj/machine_learning_examples/linear_regression_class/'

    for line in open('train.csv'):
        # print(line)
        x, y = line.split(',')
        #print(x)
        #print(y)
        X.append(float(x))
        Y.append(float(y))

    Xtest = []
    Ytest = []

    for line in open('test.csv'):
        # print(line)
        x, y = line.split(',')
        #print(x)
        #print(y)
        Xtest.append(float(x))
        Ytest.append(float(y))

    X = np.array(X)
    Y = np.array(Y)
    Xtest = np.array(Xtest)
    Ytest = np.array(Ytest)
    originalX = Xtest
    Xtest = np.vstack([np.ones(len(Xtest)), Xtest]).T



    X = scale_dataset_y(X)
    Y = scale_dataset_y(Y)

    plt.scatter(X, Y)
    plt.show()

    N = len(X)
    X = np.vstack([np.ones(N), X]).T

    # m,c = np.polyfit(X, Y, 1)
    # print(m)
    # print(c)
    D = 2
    w = np.random.randn(D) / np.sqrt(D)
    initial_theta = np.array([0.5, 0.5])
    initial_theta = w

    learning_rate = 0.001
    iterations = 1000

    theta = trainData(X, Y, initial_theta, learning_rate, iterations)

    testData(Xtest, Ytest, theta, originalX)


if __name__ == "__main__":
    main()


