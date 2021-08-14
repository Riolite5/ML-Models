import numpy as np
from util import get_data
from datetime import datetime 
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn
from future.utils import iteritems

class NaiveBayes(object):
	def fit(self,X,Y,smoothing=1e-2):
		self.gaussians = dict() #{class: {mean:mu,var:var}} 
								#-->The parameters describing the liklihood gaussian for each feature|class
		self.priors    = dict() #{class: P(C)}
		labels = set(Y)

		for c in labels:
			current_x = X[Y==c] #Features|class
			self.gaussians[c] = {
				'mean': current_x.mean(axis = 0),
				'var' : current_x.var(axis = 0) + smoothing,
			}

			self.priors[c] = float(len(Y[Y==c]))/len(Y)

	def score(self,X,Y):
		P = self.predict(X)
		return np.mean(P == Y)

	def predict(self,X):
		N, D = X.shape
		K = len(self.gaussians)
		P = np.zeros((N,K))

		for c,g in iteritems(self.gaussians):
			mean, var = g['mean'],g['var']
			P[:,c] = mvn.logpdf(X, mean=mean, cov=var) + np.log(self.priors[c]) #log liklihoods + log(prior)
		return np.argmax(P, axis=1)		




if __name__ == '__main__':
	X,Y = get_data(10000)
	Ntrain = len(Y) // 2 #number of training examples
	Xtrain, Ytrain = X[:Ntrain],Y[:Ntrain]
	Xtest, Ytest   = X[Ntrain:],Y[Ntrain:]

	model = NaiveBayes()

	t0 = datetime.now()
	model.fit(Xtrain,Ytrain)
	print(f'Training time is: {datetime.now()-t0}')

	t0 = datetime.now()
	print(f"Train Accuracy: {model.score(Xtrain,Ytrain)}")
	print(f'Time to score on the training set: {datetime.now() - t0}')
	print(f"Train set size: {len(Ytrain)}")

	t0 = datetime.now()
	print(f"Test Accuracy: {model.score(Xtest,Ytest)}")
	print(f"Time to score on the test set: {datetime.now() - t0}")
	print(f"Test set size: {len(Ytest)}")