import numpy as np
import scipy
from scipy import linalg
from sklearn.metrics import mean_squared_error
from sklearn.utils.extmath import svd_flip
from sklearn.preprocessing import MinMaxScaler

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


def poly_kernel(X, Y=None, degree=3, gamma=1, coef0=0):
    '''
    Compute the polynomial kernel between X and Y.

    K(X, Y) = (gamma <X, Y> + coef0)^{degree}
    '''
    if Y is None:
        Y = X
    K = np.dot(X, Y.T, dense_output=True)
    K *= gamma
    K += coef0
    K **= degree
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
    return K, K_fit_rows, K_fit_all

# switch the mode to either 'transform' or 'recon'


def kernel_PCA(X_train, X_test, n_components=None, kernel='rbf', gamma=None, mode='recon', alpha=1, degree=3, scaled=True):
    if scaled:
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    if kernel == 'rbf':
        K = rbf_kernel(X_train_scaled, gamma=gamma)
    elif kernel == 'poly':
        K = poly_kernel(X_train_scaled, degree=degree)
    K, K_fit_rows, K_fit_all = center_kernel(K)

    if n_components is None:
        n_components = K.shape[0]
    else:
        n_components = min(K.shape[0], n_components)

    # Solve for eigenvalues and eigenvectors, subset_by_index specifies the indices of smallest/largest to return
    eigenvalues, eigenvectors = scipy.linalg.eigh(
        K, subset_by_index=(K.shape[0] - n_components, K.shape[0]-1))

    # flip eigenvectors' sign to enforce deterministic output
    eigenvectors, _ = svd_flip(eigenvectors, np.zeros_like(eigenvectors).T)

    # Get indices of eigenvalues from largest to smallest
    indices = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]
    eigenvectors = eigenvectors[:, eigenvalues > 0]
    eigenvalues = eigenvalues[eigenvalues > 0]
    # You are supposed to find eigenvectors of K and then multiply them by K, multiplying a matrix and its eigenvector results in the same eigenvector scaled by the eigenvalue (by definition).
    X_transformed = eigenvectors * np.sqrt(eigenvalues)

    K = rbf_kernel(X_test_scaled, X_train_scaled, gamma)
    # Compute centered gram matrix between X_test and training data X_train
    K_pred_cols = (np.sum(K, axis=1) / K_fit_rows.shape[0])[:, np.newaxis]
    K -= K_fit_rows
    K -= K_pred_cols
    K += K_fit_all

    # scale eigenvectors (properly account for null-space for dot product)
    non_zeros = np.flatnonzero(eigenvalues)
    scaled_alphas = np.zeros_like(eigenvectors)
    scaled_alphas[:, non_zeros] = eigenvectors[:,
                                               non_zeros] / np.sqrt(eigenvalues[non_zeros])
    # Project with a scalar product between K and the scaled eigenvectors
    X_test_transformed = np.dot(K, scaled_alphas)
    if mode == 'transform':
        return X_test_transformed, eigenvectors, eigenvalues

    elif mode == 'recon':
        n_samples = X_transformed.shape[0]
        if kernel == 'rbf':
            K = rbf_kernel(X_transformed, gamma=gamma)
        elif kernel == 'poly':
            K = poly_kernel(X_transformed, degree=degree)
        K.flat[:: n_samples + 1] += alpha
        dual_coef = linalg.solve(
            K, X_train_scaled, assume_a="pos", overwrite_a=True)
        if kernel == 'rbf':
            K = rbf_kernel(X_test_transformed, X_transformed, gamma=gamma)
        elif kernel == 'poly':
            K = poly_kernel(X_test_transformed, X_transformed, degree=degree)
        X_test_recon = np.dot(K, dual_coef)
        if scaled:
            X_test_recon = scaler.inverse_transform(X_test_recon)
        err = mean_squared_error(X_test, X_test_recon)
        print('\n\n Own rbf kpca implementation')
        print(f'{kernel} kernel, mse err = {err}')
        return X_test_recon, err
