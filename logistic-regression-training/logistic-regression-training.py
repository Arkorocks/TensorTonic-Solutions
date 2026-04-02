import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    n_sample,n_features = X.shape
    w = np.zeros(n_features)
    b= 0.0
    for _ in range(steps):
        z = X@w + b
        p = _sigmoid(z)
        error = p-y 
        dw = (1 / n_sample) * (X.T @ error)   # shape: (n_features,)
        db = (1 / n_sample) * np.sum(error)   # scalar

        # 5. update
        w -= lr * dw
        b -= lr * db

    return w, b
    
      
    
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    pass