import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

X = []
Y = []

for line in open('data_2d.txt'):
	x1,x2,y = line.split(',')
	x1 = float(x1)
	x2 = float(x2)
	y  = float(y)
	X.append([x1,x2,1])
	Y.append(y)

# convert to numpy arrays
X = np.array(X)
Y = np.array(Y)


#plot the data
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X[:,0], X[:,1], Y)
plt.show()


#find the weights using the closed form solution
w = np.linalg.solve(np.dot(X.T,X), np.dot(X.T,Y))
Yhat = np.dot(X,w)

#R-squared
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)

print(f"R2 is: {r2}")