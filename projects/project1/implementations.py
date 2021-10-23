import numpy as np

""" Error and Loss Calculations """

def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)


def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))


def compute_loss(y, tx, w): #repetitive with calcultate_mse function
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)
    return calculate_mse(e)
    
""" Least squares, GD and SGD """
# Each function returns the optimal weights for prediction

# Implements the solution of the normal equation
def least_squares(y, tx):
    """calculate the least squares solution."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y,tx,w)
    return w, loss
    
def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        loss = calculate_mse(err)
        # update w by gradient descent
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    w = ws[-1] # last weight vector of weight matrix
    loss = losses[-1] # corresponding loss

    return w, loss
    
def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def least_squares_SGD(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = compute_loss(y, tx, w)
            # store w and loss
            ws.append(w)
            losses.append(loss)

        print("SGD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    
    w = ws[-1]
    loss = losses[-1] # corresponding lost
    
    return w, loss
    
""" Ridge regression """
# Corrects the overfitting that could happen with previous functions

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    
    # Solve the linear matrix equation to find the optimal weight vector
    w_opt = np.linalg.solve(a, b)
    
    # Compute the loss with the weight vector found
    loss = compute_loss(y,tx,w_opt)

    return w, loss
    

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

"""Logistic Regression"""

def sigmoid(t):
    """apply the sigmoid function on t."""
    sigm = 1/(1 + np.exp(-t))
    return sigm

def calculate_loss(y, tx, w):
    """compute the loss: negative log likelihood."""
    tmp = sigmoid(tx.dot(w)) #has the same shape as y
    loss = (-y.T).dot(np.log(tmp)) + ((1-y).T).dot(np.log(1-tmp))
    return loss

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)
    return grad

def calculate_hessian(y, tx, w):
    """return the Hessian of the loss function."""
    # calculate Hessian using formula derived in class
    sigm = sigmoid(tx.dot(w))
    tmp = np.diag(sigm.T[0])
    S = np.multiply(tmp,(1-tmp)) # function multiply multiplies vectors element wise (produit scalaire)
    hess = tx.T.dot(S).dot(tx)
    return hess

def logistic_regression(y, tx, w):
    """return the loss, gradient, and Hessian."""
    # return loss, gradient, and Hessian: 
    loss = calculate_loss(y, tx, w)
    grad = calculate_gradient(y, tx, w)
    hess = calculate_hessian(y, tx, w)
    return loss, grad, hess

def learning_by_newton_method(y, tx, w, gamma):
    """
    Do one step on Newton's method.
    return the loss and updated w.
    """
    # return loss, gradient and Hessian: 
    loss, grad, hess = logistic_regression(y, tx, w)
    
    # update w:
    w -= gamma*np.linalg.inv(hess).dot(grad) #cf slides its explained

    return loss, w


"""Regularized Logistic Regression"""

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient"""
    # return loss, gradient 
    loss = calculate_loss(y, tx, w) + lambda_* w.T.dot(w) # squeeze removes axes of length one from vector
    gradient = calculate_gradient(y, tx, w) + 2 * lambda_ * w 
    
    return loss, grad

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    # return loss, gradient:
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    # update w: 
    w -= gamma * gradient

    return loss, w