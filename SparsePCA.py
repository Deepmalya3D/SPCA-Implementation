import numpy as np
from tqdm import tqdm
from sklearn.linear_model import ElasticNet

def SPCA(X, n_components, lambda1, lambda2 = 0.01, max_iter = 10000, tol = 0.0001):
    """
     SPCA algorithm for simultaneous sparse coding and principal component analysis.
    
     Parameters:
         X: numpy array of shape (n_samples, n_features)
             The input data matrix.
         n_components: int
             Number of principal components to retain.
         lambda1: float
             Sparsity controlling parameter. Higher values lead to sparser components.
         lambda2: float
             Amount of ridge shrinkage to apply in order to improve conditioning when calling the transform method.
         
         max_iter: int
             Maximum number of iterations for the optimization.
         tol: float
             Maximum allowed tolerance
    
     Returns:
         sparse_components : numpy array of shape (n_samples, n_components)
             The sparse codes for the input data.
         loadings: numpy array of shape (n_features, n_components)
             The projection matrix for the principal components.
    """

    cor_mat = X.T@X
    _, V = np.linalg.eig(cor_mat)
    V = V[_.argsort()[::-1]]
    A = V[:, :n_components]
    B = np.zeros((V.shape[0], n_components))

    iter = 0
    pbar = tqdm(total = max_iter+1)
    while iter < max_iter:
        diff = 0
        B_temp = np.zeros((V.shape[0], n_components))

        for j in range(A.shape[1]):
            y_s = X@A[:,j:j+1]
            x_s = X
            alpha = lambda1[j] + 2*lambda2
            l1_ratio = lambda1[j]/(lambda1[j] + 2*lambda2)

            elas_net = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
            elas_net.fit(x_s*(np.sqrt(V.shape[0]*2)), y_s*(np.sqrt(V.shape[0]*2)))
            B_temp[:,j] = np.array(elas_net.coef_)

        diff = np.linalg.norm(B_temp - B)
        B = B_temp

        cor_B = cor_mat @ B 
        U, D, V_ = np.linalg.svd(cor_B, full_matrices=False)
        A = U @ V_.T

        iter += 1
        if diff < tol:
            break
        elif iter == max_iter-1:
            print("Max Iterations reached")

        pbar.update(1)
    
    loadings = B/np.linalg.norm(B)
    sparse_components = X @ loadings
    return sparse_components, loadings
