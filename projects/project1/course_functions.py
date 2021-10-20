import numpy as np

"""Polynomial regression"""

# Feature vector extension by adding polynomial basis  
def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    feature = np.ones((len(x), 1)) #corresponds to the first column full of ones of polynomial feature expansion
    for d in range(1,degree+1) :
        feature = np.c_[feature, np.power(x, d)]
    
    return feature

"""Split the dataset into a test and training set based on the given ratio"""

# Splits the data into a test and training set based on a given ratio
def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, x_te, y_tr, y_te

"""Cross validation"""

# Returns the each group used for cross validation, the number of groups depends on parameter k_fold
def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold) # gives the number of data points in each group
    np.random.seed(seed)
    indices = np.random.permutation(num_row) # puts the numbers 0 to 49 into a different order (y is of size 50), so shuffles indices 
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    # k_indices lists what is in each group for cross validation : k_fold gives the number of groups
    # One of these groups should be used for testig and the k_fold-1 others for training
    
    return np.array(k_indices)

from costs import compute_mse
from ridge_regression import ridge_regression
from build_polynomial import build_poly

# Returns the training and test rmse for one iteration of the cross validation (the kth one)
# Cross validation to find the optimal weights for polynomial regression
# Need to use the function in a for loop : for k in range(k_fold), to use each group once as a test set
def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
   
    # get k'th subgroup in test, others in train:
    test_index = k_indices[k]
    x_te = x[test_index]
    x_tr = np.delete(x, test_index)
    y_te = y[test_index]
    y_tr = np.delete(y,test_index)

    # form data with polynomial degree: TODO
    tx_tr = build_poly(x_tr, degree)
    tx_te = build_poly(x_te, degree)

    # ridge regression: 
    weights = ridge_regression(y_tr, tx_tr, lambda_)
        
    loss_tr = np.sqrt(2 * compute_mse(y_tr, tx_tr, weights))
    loss_te = np.sqrt(2 * compute_mse(y_te, tx_te, weights))

    return loss_tr, loss_te



