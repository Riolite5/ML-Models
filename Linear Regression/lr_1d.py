import numpy as np
import matplotlib.pyplot as plt

X = []
Y = []


#read x,y 
for line in open('data_1d.txt'):
	x, y = line.split(',')
	X.append(float(x))
	Y.append(float(y))

X = np.array(X)
Y = np.array(Y)


# scatter x and y
plt.scatter(X,Y)
plt.show()

# closed formal normal equation to calculate a and b
denominator = X.dot(X) - X.mean() * X.sum()

a = ((X.dot(Y)) - Y.mean()*X.sum())/ denominator
b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y)) / denominator

# let's caclculate predicted Y
Yhat = a*X + b

#plot all
plt.scatter(X,Y)
plt.plot(X,Yhat)
plt.show()

#calculate r-squared
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)

print(f"R2 is: {r2}")