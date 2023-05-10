import numpy as np
import scipy

# Get the square of l2 norm


def l2(X, Y):
    l2_arr = np.empty([X.shape[0], Y.shape[0]])
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            diff = X[i, :]-Y[j, :]
            l2_arr[i, j] = np.dot(diff, diff)
    return l2_arr

# Get rbf kernel


def rbf_kernel(X, Y=None, gamma=None):
    '''
    Compute the rbf (gaussian) kernel between X and Y.

        K(x, y) = exp(-gamma ||x-y||^2)

    for each pair of rows x in X and y in Y.
    '''
    if Y is None:
        Y = X
    K = np.exp(-1*gamma*l2(X, Y))
    return K

# Center the kernel matrix


def center_kernel(K):
    n_samples = K.shape[0]
    K_fit_rows = np.sum(K, axis=0) / n_samples
    K_fit_all = K_fit_rows.sum() / n_samples
    K_pred_cols = (np.sum(K, axis=1) / K_fit_rows.shape[0])[:, np.newaxis]
    K -= K_fit_rows
    K -= K_pred_cols
    K += K_fit_all
    return K


def kernel_PCA(X, n_components=None, kernel='rbf', gamma=None):
    if kernel == 'rbf':
        K = rbf_kernel(X, gamma=gamma)
        K = center_kernel(K)
    if n_components is None:
        n_components = K.shape[0]

    # Solve for eigenvalues and eigenvectors, subset_by_index specifies the indices of smallest/largest to return
    eigenvalues, eigenvectors = scipy.linalg.eigh(
        K, subset_by_index=(K.shape[0] - n_components, K.shape[0]-1))

    # Get indices of eigenvalues from largest to smallest
    indices = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]
    eigenvectors = eigenvectors[:, eigenvalues > 0]
    eigenvalues = eigenvalues[eigenvalues > 0]
    # You are supposed to find eigenvectors of K and then multiply them by K, multiplying a matrix and its eigenvector results in the same eigenvector scaled by the eigenvalue (by definition).
    X_transformed = eigenvectors * np.sqrt(eigenvalues)
    return X_transformed, eigenvectors, eigenvalues
