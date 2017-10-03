# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017

"""
from __future__ import print_function
import matplotlib.pyplot as plt
from numpy import *
import numpy as np
from sklearn.datasets import load_boston
from scipy.misc import logsumexp

np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506, 1)), x), axis=1)  # add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))


# helper function
def l2(A, B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
    B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
    dist = A_norm + B_norm - 2 * A.dot(B.transpose())
    return dist



#helper function
def run_on_fold(x_test, y_test, x_train, y_train, taus):
    '''
    Input: x_test is the N_test x d design matrix
           y_test is the N_test x 1 targets vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           taus is a vector of tau values to evaluate
    output: losses a vector of average losses one for each tau value
    '''
    N_test = x_test.shape[0]
    losses = np.zeros(taus.shape)
    for j,tau in enumerate(taus):
        predictions =  np.array([LRLS(x_test[i,:].reshape(d,1),x_train,y_train, tau) \
                        for i in range(N_test)])
        losses[j] = ((predictions.flatten()-y_test.flatten())**2).mean()
    return losses


def compute_A(test_datum, x_train, tau):
    norms = l2(x_train, np.array(test_datum.reshape(1,14)))
    distance = np.exp(-norms/2/(tau**2))
    total_distance = np.exp(logsumexp(-norms/2/(tau**2)))
    A = distance/total_distance
    where_are_NaNs = isnan(A)
    A[where_are_NaNs] = 0
    #  print(test_datum)
    #  print(tau)
    # print(total_distance)
    #  print(x_train)
    # print(norms)
    # exit(0)
    return np.diag(A.reshape(1,A.shape[0])[0])



# to implement
def LRLS(test_datum, x_train, y_train, tau, lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    A = compute_A(test_datum,x_train,tau)
    # print(np.dot(x_train.transpose(), A).dot(x_train).shape)
    # print(x_train.transpose().dot(A).shape)
    # print(A.shape)
    # print(y_train.shape)
    # print((lam*(np.identity(y_train.shape[1]))))
    w = np.linalg.solve(
        np.dot(x_train.transpose(),A).dot(x_train)
        + lam*(np.identity(x_train.shape[1])),
        (x_train.transpose()).dot(A).dot(y_train)
    )
    # print(A)
    # print(w)
    return np.dot(w.transpose(),test_datum)


def run_k_fold(x, y, taus, k):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector
           taus is a vector of tau values to evaluate
           K in the number of folds
    output is losses a vector of k-fold cross validation losses one for each tau value
    '''
    sample_size = int(x.shape[0]/k)
    loss = []
    for i in range(k):
        train_idex_list = [
            j for j in range(x.shape[0])
            if j <sample_size*i or j>=sample_size*(i+1)]
        temp_loss = run_on_fold(
            x[sample_size*i:sample_size*(i+1)],
            y[sample_size*i:sample_size*(i+1)],
            x[train_idex_list],
            y[train_idex_list],
            taus
        )
        loss.append(temp_loss)
    return np.array(loss)



if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0, 3, 200)
    losses = run_k_fold(x, y, taus, k=5)
    plt.plot(taus,losses.transpose())
    plt.show()
    print("min loss = {}".format(losses.min()))
