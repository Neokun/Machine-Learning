import numpy as np

def linear_loss_naive(W, X, y, reg):
    """
    Linear loss function, naive implementation (with loops)

    Inputs have dimension D, there are N examples.

    Inputs:
    - W: A numpy array of shape (D, 1) containing weights.
    - X: A numpy array of shape (N, D) containing data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where c is a real number.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    N, D = X.shape
    y = y.reshape((N, 1))
    W = W.reshape((D, 1))
    #if reg > 1: reg = 1/reg

    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the linear loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    # Get loss
    y_pred = np.zeros(N)
    for i in range(N):
        for j in range(D):
            y_pred[i] += X[i][j]*W[j]
        loss += (y_pred[i] - y[i])**2

    loss /= 2*N
    loss += 0.5*reg*np.sum(W*W)

    # Get gradient
    for j in range(D):
        for i in range(N):
            dW[j] += X[i][j]*(y_pred[i] - y[i])

    dW /= N
    dW += reg*W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    return loss, dW


def linear_loss_vectorized(W, X, y, reg):
    """
    Linear loss function, vectorized version.

    Inputs and outputs are the same as linear_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    N, D = X.shape
    y = y.reshape((N, 1))
    W = W.reshape((D, 1))
    #if reg > 1: reg = 1/reg

    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the linear loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    
    y_pred = np.dot(X, W)
    loss = np.linalg.norm(y_pred - y)**2/(2*N)
    loss += 0.5*reg*np.sum(W*W)
    
    dW = np.dot(X.T, (y_pred - y))
    dW /= N
    dW += reg*W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    return loss, dW