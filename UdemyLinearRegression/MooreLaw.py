import re
import numpy as np
import matplotlib.pyplot as plt

X = []
Y = []

non_decimal = re.compile(r'[^\d]+')
path = '/Users/Pankaj/machine_learning_examples/linear_regression_class/'

for line in open(path + 'moore.csv'):
    r = line.split('\t')

    x = int(non_decimal.sub('', r[2].split('[')[0] ))
    y = int(non_decimal.sub('', r[1].split('[')[0] ))
    X.append(x)
    Y.append(y)


X = np.array(X)
Y = np.array(Y)


Y = np.log(Y)
plt.scatter(X, Y)
#plt.show()

#find the mean and y-intercept
m,c = np.polyfit(X,Y,1)


Yhat = m * X + c

plt.scatter(X,Y)
plt.plot(X, Yhat)
plt.show()

d1 = Y - Yhat
d2 = Y - np.mean(Y)

r2 = 1 - (d1.dot(d1) / d2.dot(d2))
print (r2)

