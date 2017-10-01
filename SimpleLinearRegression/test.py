
import numpy as np
import pandas as pd

I = np.eye(3)
Q = np.array([
    [0,3/4.0,0],
    [1/2.0,0,1/2.0], [0, 3/4.0, 0]]
    )

sub = I - Q

#print(np.linalg.inv(sub))

# X = []
# Y = []
#
# for line in open('test.csv'):
#     x,y = line.split(',')
#     X.append(float(x))
#     Y.append(float(y))
#
# print(X)
# print(Y)


print(Q)
print(Q[0])
print(Q[0:2])
print(Q[0][1])
print(Q[0, 1:3])
