# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 12:30:54 2019

@author: muli2487
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
## (1) Data preparation
df=pd.read_csv('winequality-white.csv', sep = ';')
df
X = df.values[:, :11]
Y = df.values[:, 11]
print('Data shape:', 'X:', X.shape, 'Y:', Y.shape)
# data normalization
min_vals = np.min(X, axis = 0)
max_vals = np.max(X, axis = 0)
X1 = (X-min_vals)/(max_vals-min_vals)
X_train, X_test, Y_train, Y_test = train_test_split(X1, Y, test_size=0.33, random_state=42)
##(2) Assume a linear mode that y = w0*1 + w_1*x_1 +w_2*x_2+...+ w_11*x_11
def predict(X, w):
    '''
    X: input feature vectors:m*n
    w: weights
    
    return Y_hat
    '''
    # Prediction
    Y_hat = np.zeros((X.shape[0]))
    for idx, x in enumerate(X):          
        y_hat = w[0] + np.dot(w[1:].T, np.c_[x]) # linear model
        Y_hat[idx] = y_hat    
    return Y_hat

## (3) Loss function: L = 1/2 * sum(y_hat_i - y_i)^2
def loss(w, X, Y):
    '''
    w: weights
    X: input feature vectors
    Y: targets
    '''
    Y_hat = predict(X, w)
    loss = 1/2* np.sum(np.square(Y - Y_hat))
    
    return loss

# Optimization: Gradient Descent
def GD(X1, Y, lr = 0.001, delta = 0.01, max_iter = 100):
    '''
    X: training data
    Y: training target
    lr: learning rate
    max_iter: the max iterations
    '''
    
    m = len(Y)
    b = np.reshape(Y, [Y.shape[0],1])
    w = np.random.rand(X1.shape[1] + 1, 1)
    A = np.c_[np.ones((m, 1)), X1]
    gradient = A.T.dot(np.dot(A, w)-b)
    
    loss_hist = np.zeros(max_iter) # history of loss
    w_hist = np.zeros((max_iter, w.shape[0])) # history of weight
    loss_w = 0
    i = 0                  
    while(np.linalg.norm(gradient) > delta) and (i < max_iter):
        w_hist[i,:] = w.T
        loss_w = loss(w, X, Y)
        print(i, 'loss:', loss_w)
        loss_hist[i] = loss_w
        
        w = w - lr*gradient        
        gradient = A.T.dot(np.dot(A, w)-b) # update the gradient using new w
        i = i + 1
        
    w_star = w  
    return w_star, loss_hist, w_hist
# example
w_star, loss_hist, w_hist = GD(X1, Y, lr = 0.0001, delta = 0.01, max_iter = 10)

# show the Loss curve
from matplotlib import pyplot as plt
plt.plot(range(10), loss_hist)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()

# function to create a list containing mini-batches 
def create_mini_batches(X1, Y, batch_size): 
    mini_batches = [] 
    data = np.hstack((X1, Y)) 
    np.random.shuffle(data) 
    n_minibatches = data.shape[0] // batch_size 
    i = 0
  
    for i in range(n_minibatches + 1): 
        mini_batch = data[i * batch_size:(i + 1)*batch_size, :] 
        X_mini = mini_batch[:, :-1] 
        Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
        mini_batches.append((X_mini, Y_mini)) 
    if data.shape[0] % batch_size != 0: 
        mini_batch = data[i * batch_size:data.shape[0]] 
        X_mini = mini_batch[:, :-1] 
        Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
        mini_batches.append((X_mini, Y_mini)) 
    return mini_batches 
# Optimization: implement the minibatch Gradient Descent approach
def SGD(X1, Y, lr = 0.001, batch_size = 32, epoch = 100):
    '''
    X: training data
    Y: training target
    lr: learning rate
    batch_size: batch size
    epoch: number of max epoches
    
    return: w_star, loss_hist, w_hist
    '''
    
    m = len(Y)
    w = np.random.rand(X.shape[1] + 1, 1)
    loss_hist = np.zeros(epoch)
    w_hist = np.zeros((epoch, w.shape[0]))
    #t = 0 

    #Your code here:
    for i in range(epoch):
        #shuffle the data
        shuffled_indices = np.random.permutation(m)
        X_b_shuffled = X_train[shuffled_indices]
        y_shuffled = Y_train[shuffled_indices]
        
        
        for b in range(int(m/batch_size)):
             #t += 1
             xi = X_b_shuffled[i:i+batch_size]
             yi = y_shuffled[i:i+batch_size]
             gradients = m/batch_size * xi.T.dot(xi.dot(w) - yi)
             #eta = learning_schedule(lr)
             w  = w - lr * gradients
            
            
        
   

        w_hist[i,:] = w.T
        print(i, loss_hist[i])
        
    w_star = w  
    return w_star, loss_hist, w_hist

# example
w_star, loss_hist, w_hist = SGD(X_test, Y_test, lr = 0.0001, batch_size = 32, epoch = 100)

# show the Loss curve
from matplotlib import pyplot as plt
plt.plot(range(10), loss_hist)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()

# Measure performance

import sklearn.metrics as sm



print "Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2) 

print "Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2) 

print "Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2) 

print "Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2) 