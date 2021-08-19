from future.utils import iteritems

import numpy as np
import matplotlib.pyplot as plt
from sortedcontainers import SortedList
from util import get_data
from datetime import datetime


class KNN(object):
	def __init__(self,k):
		self.k = k

	def fit(self,X,y):
		self.X = X
		self.y = y

	def predict(self,X):
		y = np.zeros(len(X))

		for i,x in enumerate(X):
			sl = SortedList()
			for j,xt in enumerate(X): #test points
				diff = x-xt
				d = diff.dot(diff)
				if (len(sl) < self.k): #the sorted list has less elements than the number of neighbours
					sl.add((d,self.y[j])) 
				else:
					if d< sl[-1][0]:
						del sl[-1]
						sl.add((d,self.y[j]))

			votes = {}

			for _,v in sl:
				votes[v] = votes.get(v,0) + 1

			max_votes = 0
			max_votes_class = -1

			for v,count in iteritems(votes):
				if count>max_votes:
					max_votes = count
					max_votes_class = v
			y[i] = max_votes_class

		return y




	def score(self,X,Y):
		P = self.predict(X)
		return np.mean(P==Y)


if __name__ == '__main__':
	X,Y = get_data(2000)
	Ntrain = 1000
	Xtrain, Ytrain = X[:Ntrain],Y[:Ntrain]
	Xtest, Ytest   = X[Ntrain:],Y[:Ntrain]
	train_scores = []
	test_scores  = []
	ks = (1,2,3,4,5)

	for k in ks:
		print(f'\nk = {k}')
		knn = KNN(k)

		t0 = datetime.now()
		knn.fit(Xtrain,Ytrain)
		print(f"Training time: {datetime.now() - t0}")

		t0 = datetime.now()
		train_score = knn.score(Xtrain,Ytrain)
		train_scores.append(train_score)
		print(f"Train accuracy: {train_score}")
		print(f"Time to compute train score: {datetime.now() - t0}")

		t0 = datetime.now()
		test_score = knn.score(Xtest,Ytest)
		test_scores.append(test_score)
		print(f"Testing accuracy: {test_score}")
		print(f"Time to compute test score: {datetime.now() - t0}")

	plt.plot(ks,train_scores, label = 'train scores')
	plt.plot(ks,test_scores, label = 'test scores')
	plt.legend()
	plt.show()

