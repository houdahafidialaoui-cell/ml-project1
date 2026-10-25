"""Core ML methods for Project 1 (only NumPy allowed).

All functions return (w, loss) where w is the LAST weight vector (1D array)
and loss is the cost at that w. For the regularized methods the returned
loss does NOT include the penalty term.
"""
import numpy as np

#The following functions will help us implement the required methods. Note to myself :  Before submission , make sure they're used , if not just remove them.

def compute_mse(y, tx, w):
    """MSE loss with the 0.5 factor to be consistent with lecture notes: 1/(2N) * ||y - tx w||^2."""
    e = y - tx @ w
    return 0.5 * np.mean(e ** 2)


def compute_mse_gradient(y, tx, w):
    """Gradient of the MSE loss: -1/N * tx^T (y - tx w)."""
    e = y - tx @ w
    return -tx.T @ e / len(y)


def sigmoid(z):
    """Sigmoid function: turns a score into a probability between 0 and 1."""
    return 1 / (1 + np.exp(-z))


def compute_logistic_loss(y, tx, w):
    """Mean logistic loss: 1/N * sum(log(1 + exp(tx w)) - y * tx w)."""
    z = tx @ w
    return np.mean(np.logaddexp(0, z) - y * z) # logaddexp(0, z) is a numerically stable way to compute log(1 + exp(z)) , because exp(z) can be very large for large z, leading to overflow errors.


def compute_logistic_gradient(y, tx, w):
    """Gradient of the mean logistic loss: 1/N * tx^T (sigmoid(tx w) - y)."""
    z = tx @ w
    return tx.T @ (sigmoid(z) - y) / len(y)


# The following are the Required functions : 

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using full-batch gradient descent.
    Args:
        y:         labels, shape (N,).
        tx:        data matrix, shape (N, D).
        initial_w: starting weight vector, shape (D,).
        max_iters: max number of gradient steps to run.
        gamma:     step size (learning rate).
    Returns:
        w:    the last weight vector, shape (D,).
        loss: the MSE loss at w.
    """
    w = np.array(initial_w, dtype=float)
    for _ in range(max_iters):
        w = w - gamma * compute_mse_gradient(y, tx, w) # Full-batch gradient descent update step
    return w, compute_mse(y, tx, w)
 
 
def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent (batch size 1).
    Args:
        y:         labels, shape (N,).
        tx:        data matrix, shape (N, D).
        initial_w: starting weight vector, shape (D,).
        max_iters: number of SGD steps; each step uses ONE random datapoint.
        gamma:     step size (learning rate).
    Returns:
        w:    the last weight vector, shape (D,).
        loss: the MSE loss at w, computed on the full dataset.
    """
    w = np.array(initial_w, dtype=float)
    n = len(y)
    for _ in range(max_iters):
        i = np.random.randint(n)  # pick a single random datapoint (Question: Should we use a random seed for reproducibility?)
        grad = compute_mse_gradient(y[i:i + 1], tx[i:i + 1], w) 
        w = w - gamma * grad
    return w, compute_mse(y, tx, w)
 
 
def least_squares(y, tx):
    """Least squares regression using the normal equations.
 
    Solves (tx^T tx) w = tx^T y.
 
    Args:
        y:  labels, shape (N,).
        tx: data matrix, shape (N, D).
 
    Returns:
        w:    the optimal weight vector, shape (D,).
        loss: the MSE loss at w.
    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return w, compute_mse(y, tx, w)
 
 
def ridge_regression(y, tx, lambda_):
    """Ridge regression using the normal equations.
 
    Penalty is lambda_ * ||w||^2 (course convention), which gives
    w = (tx^T tx + 2 N lambda_ I)^{-1} tx^T y.
 
    Args:
        y:       labels, shape (N,).
        tx:      data matrix, shape (N, D).
        lambda_: regularization strength (>= 0). Larger = smaller weights.
 
    Returns:
        w:    the weight vector, shape (D,).
        loss: the MSE loss at w, WITHOUT the penalty term.
    """
    n, d = tx.shape
    a = tx.T @ tx + 2 * n * lambda_ * np.eye(d)
    w = np.linalg.solve(a, tx.T @ y)
    return w, compute_mse(y, tx, w)
 
 
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression (y in {0, 1}) using gradient descent.
 
    Args:
        y:         labels in {0, 1}, shape (N,).
        tx:        data matrix, shape (N, D).
        initial_w: starting weight vector, shape (D,).
        max_iters: number of gradient steps to run.
        gamma:     step size (learning rate).
 
    Returns:
        w:    the last weight vector, shape (D,).
        loss: the logistic loss at w.
    """
    w = np.array(initial_w, dtype=float)
    for _ in range(max_iters):
        w = w - gamma * compute_logistic_gradient(y, tx, w)
    return w, compute_logistic_loss(y, tx, w)
 
 
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """L2-regularized logistic regression using gradient descent.
 
    Penalty is lambda_ * ||w||^2, so its gradient is 2 * lambda_ * w.
 
    Args:
        y:         labels in {0, 1}, shape (N,).
        tx:        data matrix, shape (N, D).
        lambda_:   regularization strength (>= 0).
        initial_w: starting weight vector, shape (D,).
        max_iters: number of gradient steps to run.
        gamma:     step size (learning rate).
 
    Returns:
        w:    the last weight vector, shape (D,).
        loss: the logistic loss at w, WITHOUT the penalty term.
    """
    w = np.array(initial_w, dtype=float)
    for _ in range(max_iters):
        grad = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * grad
    return w, compute_logistic_loss(y, tx, w)
 