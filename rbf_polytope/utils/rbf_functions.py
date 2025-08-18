import numpy as np

def gaussian_rbf(theta:np.ndarray, 
                 centers:np.ndarray, 
                 widths:np.ndarray)->np.ndarray:
    distances = np.linalg.norm(centers - theta.flatten(), axis=1)
    weights = np.exp(- widths * distances**2)
    return weights / np.sum(weights)

