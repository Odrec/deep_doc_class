# -*- coding: utf-8 -*-

#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.

import numpy as np
import sklearn.preprocessing

class TSNE:
	
	_X = np.array([])
	_max_iter = 0
	_perplexity = 0
	_no_dims = 0
	_initial_dims = 0
	
	# tolerance
	_tol = 1e-5
	# momentum
	_initial_momentum = 0.5;
	_final_momentum = 0.8;
	# learning rate
	_eta = 500;
	_min_gain = 0.01;
	
	
	def __init__(self, X, no_dims=3, initial_dims=50, max_iter=300, perplexity=30.0):
		self._X = X
		self._no_dims = no_dims
		self._initial_dims = initial_dims
		self._max_iter = max_iter
		self._perplexity = perplexity

	def get_x(self):
		return self.__X


	def get_max_iter(self):
	    return self.__max_iter


	def get_perplexity(self):
	    return self.__perplexity


	def get_no_dims(self):
	    return self.__no_dims


	def get_initial_dims(self):
	    return self.__initial_dims


	def get_tol(self):
	    return self.__tol


	def get_initial_momentum(self):
	    return self.__initial_momentum


	def get_final_momentum(self):
	    return self.__final_momentum


	def get_eta(self):
	    return self.__eta


	def get_min_gain(self):
	    return self.__min_gain


	def set_x(self, value):
	    self.__X = value


	def set_max_iter(self, value):
	    self.__max_iter = value


	def set_perplexity(self, value):
	    self.__perplexity = value


	def set_no_dims(self, value):
	    self.__no_dims = value


	def set_initial_dims(self, value):
	    self.__initial_dims = value


	def set_tol(self, value):
	    self.__tol = value


	def set_initial_momentum(self, value):
	    self.__initial_momentum = value


	def set_final_momentum(self, value):
	    self.__final_momentum = value


	def set_eta(self, value):
	    self.__eta = value


	def set_min_gain(self, value):
	    self.__min_gain = value

	
	def Hbeta(self, D = np.array([]), beta = 1.0):
		""" compute P and H
		H :=Entropy for the point x depending on beta = 1/(2*sigma^2) (the Gaussian Kernel)
		P :=Probabilities for picking y given x depending on euclidean D(istances)
		List of absolute probability """
		P = np.exp(-D.copy() * beta);
		# Sum over all single probabilities -> p(x)
		sumP = sum(P);
		# Entropy for resulting for the current beta
		H = np.log(sumP) + beta * np.sum(D * P) / sumP;
		# List of normed probabilities for each y to belong to x given that x -> p(y|x)
		P = P / sumP;
	# 	esnUtil.checkDataSet(P)
		return H, P;
		
		
	def x2p(self):
		# binary search for finding the right gaussian kernel for each point
		# meaning the entropy is near to the given perplexity for each point
	
		# Initialize some variables
		print("Computing pairwise distances...")
		(n, d) = self._X.shape;
		# square every single value
		sum_X = np.sum(np.square(self._X), 1);
		# distance Matrix D (-2x*y+x*x+y*y = euc(x,y)^2)
		D = np.add(np.add(-2 * np.dot(self._X, self._X.T), sum_X).T, sum_X);
# 		sklearn.preprocessing.scale(D, axis=0, with_mean=True, with_std=True, copy=False)
		# Probability matrix for each point y belonging to point x
		P = np.zeros((n, n));
		# Current gaussian parameter for each point (1/(2*sigma^2))
		beta = np.ones((n, 1));
		# Desired Entropy for each point
		logU = np.log(self._perplexity);
		
		# Loop over all datapoints
		for i in range(n):
		
			# Print progress
			if i % 500 == 0:
				print("Computing P-values for point ", i, " of ", n, "...")
		
			# Set boundaries for the binary search
			betamin = -np.inf; 
			betamax =  np.inf;
			# Take D(istances) in row i except position i itself
			Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))];
			# Compute the Gaussian kernel and entropy for the current precision
			(H, thisP) = self.Hbeta(Di, beta[i]);
				
			# Evaluate whether the perplexity is within tolerance
			Hdiff = H - logU;
			tries = 0;
			while np.abs(Hdiff) > self._tol and tries < 50:
					
				# If entropy is too high increase beta (-> lower probablities
				# since higher beta means lower sigma)
				if Hdiff > 0:
					# set new lower bound
					betamin = beta[i].copy();
					# if no upper bound was found before double beta
					if betamax == np.inf or betamax == -np.inf:
						beta[i] = beta[i] * 2;
					# else search between bounds
					else:
						beta[i] = (beta[i] + betamax) / 2;
				# Do in reverse fashion for low entropy
				else:
					betamax = beta[i].copy();
					if betamin == np.inf or betamin == -np.inf:
						beta[i] = beta[i] / 2;
					else:
						beta[i] = (beta[i] + betamin) / 2;
				
				# Recompute Entropy and probabilities
				(H, thisP) = self.Hbeta(Di, beta[i]);
				Hdiff = H - logU;
				tries = tries + 1;
				
			# Set the finally resulting row of Probability values
			P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP;
		
		return P
	
	
	def pca(self):
	
		print ("Preprocessing the data using PCA")
		(n, d) = self._X.shape;
		# substract mean of each datapoint
		X = self._X - np.tile(np.mean(self._X, 0), (n, 1));
		# get the eigenvectors and eigenvalues of the cov matrix
		(l, M) = np.linalg.eig(np.dot(X.T, X)/(n-1));
		
		# sort the eigenvectors respective to their eigenvalues
		pairs = []
		for x in range(0,len(l)):
			pairs.insert(x, (np.abs(l[x]),  np.real(M[:,x])))
		# sort the list starting with the highest eigenvalue
		pairs.sort(key=lambda tup: tup[0], reverse=True)
		
		# append the best eigenvectors to a matrix until desired dimension is reached
		M = np.hstack((pairs[i][1].reshape(d,1) for i in range(0,self._no_dims)))
		# transpose the data to these dimensions
		Y = np.dot(X, M);
		return Y;
	
	
	def run(self):
		"""Runs t-SNE on the dataset in the NxD array X to reduce its dimensionality to no_dims dimensions.
		The syntaxis of the function is Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array."""
		
		# Initialize variables
		(rows, collumns) = np.shape(self._X)
		if(collumns>self._initial_dims):
			self._X = self.pca().real;
		(n, d) = self._X.shape;
		# low dimension level values(result)
		Y = np.random.randn(n, self._no_dims);
		# low value update direction(gradient)
		dY = np.zeros((n, self._no_dims));
		# update from last iteration for regulating acceleration
		iY = np.zeros((n, self._no_dims));
		gains = np.ones((n, self._no_dims));
		
		# Compute P-values
		P = self.x2p();
		# improve p-value formular
		
		# compute (p(x|y)+p(y|x))/2n (sum p[i,:]=1)
		P = P + P.T
		P = P / np.sum(P);
		
		# initially exaggerate probabilities
		P = P * 4;
		P = np.maximum(P, 1e-12);
		sum_Y = np.zeros(n)
		num = np.zeros((n,n))
		Q = np.zeros((n,n))
		PQ = np.zeros((n,n))
		C = 0
		# Run iterations
		for iter in range(self._max_iter):
			
			# Compute pairwise affinities
			
			# square all values
			sum_Y = np.sum(np.square(Y), 1);
			# 1/1+distance roughly same as e^-d	(-> t-Distribution)	
			num = 1 / (1 + np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y));
			# set diagonals to 0 -> affinity to itself
			num[range(n), range(n)] = 0;
			# norm affinity at sum of all affinities (improved formula)
			Q = num / np.sum(num);
			# minimal value is 1/10^12
			Q = np.maximum(Q, 1e-12);
			
			# Compute gradient
			PQ = P - Q;
			for i in range(n):
				dY[i,:] = np.sum(np.tile(PQ[:,i] * num[:,i], (self._no_dims, 1)).T * (Y[i,:] - Y), 0);
				
			# Perform the update
			# choose momentum dependent on stage of the algorithm
			if iter < 20:
				momentum = self._initial_momentum
			else:
				momentum = self._final_momentum
			# increase gain if change keeps direction, decrease otherwise
			gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0));
			# ensure a minimum change
			gains[gains < self._min_gain] = self._min_gain;
			# new change is momentum times old change + new change
			iY = momentum * iY - self._eta * (gains * dY);
			# update values
			Y = Y + iY;
			# substract the mean
			Y = Y - np.tile(np.mean(Y, 0), (n, 1));
			
			# Compute current value of cost function
			if (((iter+1)%100==0) or (iter==0)):
				C = np.sum(P * np.log(P / Q));
				print ("Iteration ", (iter + 1), ": error is ", C)
				
			# adjust probabilities to their actual values after the initial phase
			if iter == 100:
				P = P / 4;
				
		# Return solution
		return Y;