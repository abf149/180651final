
from sklearn.decomposition import KernelPCA
import math
# Autoencoder imports
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from matplotlib import pyplot as plt

import numpy as np
import os
import sys
import sklearn
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA, SparsePCA
from sklearn.datasets import make_swiss_roll
from sklearn.metrics import mean_squared_error
import math

import numpy as np
from sklearn.decomposition import NMF
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


# TODO: hyperparameter search

# SparsePCA(n_components=k, alpha=1, ridge_alpha=0.01, max_iter=1000, tol=1e-08, method='lars', n_jobs=1, U_init=None, V_init=None, verbose=False, random_state=None)

def spca_exp(X_train, X_test, k):
    alpha_vals = [0.001, 0.002, 0.005, 0.01, 0.02,
                  0.05, 0.10, 0.20, 0.50, 1.0, 2.0, 5.0]
    ridge_alpha = 0.0

    best_spca_alpha = 0.0
    best_spca_alpha_err = math.inf
    for idx in range(len(alpha_vals)):
        alpha = alpha_vals[idx]
        spca = SparsePCA(n_components=k, ridge_alpha=ridge_alpha, alpha=alpha)
        spca.fit(X_train)
        X_test_reduced = spca.transform(X_test)
        X_test_recon = spca.inverse_transform(X_test_reduced)
        spca_err = mean_squared_error(X_test, X_test_recon)
        print("alpha=", alpha, "err=", spca_err)
        if spca_err < best_spca_alpha_err:
            best_spca_alpha_err = spca_err
            best_spca_alpha = alpha

    print("\n\nBest SPCA:")
    print("- alpha:", best_spca_alpha)
    print("- err:", best_spca_alpha_err)

    return best_spca_alpha, best_spca_alpha_err


def spca_exp_faces(X_train, X_test, k, h, w):
    # h is the height of each face, w is the width of each face
    alpha_vals = [0.001, 0.002, 0.005, 0.01, 0.02]
    ridge_alpha = 0.0

    best_spca_alpha = 0.0
    best_spca_alpha_err = math.inf
    for idx in range(len(alpha_vals)):
        alpha = alpha_vals[idx]
        spca = SparsePCA(n_components=k, ridge_alpha=ridge_alpha, alpha=alpha)
        spca.fit(X_train)
        X_test_reduced = spca.transform(X_test)
        X_test_recon = spca.inverse_transform(X_test_reduced)
        spca_err = mean_squared_error(X_test, X_test_recon)
        print("alpha=", alpha, "err=", spca_err)
        if spca_err < best_spca_alpha_err:
            best_spca_alpha_err = spca_err
            best_spca_alpha = alpha
            best_eigenfaces_spca = spca.components_.reshape((k, h, w))

    print("\n\nBest SPCA:")
    print("- alpha:", best_spca_alpha)
    print("- err:", best_spca_alpha_err)

    return best_spca_alpha, best_spca_alpha_err, best_eigenfaces_spca


# Drawing on inspiration from https://docs.google.com/document/d/1lmaQowAhgf1OLbnSjl3X20j1idh-BdaQPZTrJslSJc0/edit "Kernel PCA" section
# TODO: hyperparameter sweep for kernel, gamma, coef0, other attributes


def kernel_exp(X_train, X_test, k):

    lin_pca = KernelPCA(n_components=k, kernel="linear",
                        fit_inverse_transform=True)
    rbf_pca = KernelPCA(n_components=k, kernel="rbf",
                        gamma=0.0433, fit_inverse_transform=True)
    sig_pca = KernelPCA(n_components=k, kernel="sigmoid",
                        gamma=0.001, coef0=1, fit_inverse_transform=True)

    kernel_options = ((131, lin_pca, "Linear kernel"), (132, rbf_pca, "RBF kernel, $\gamma=0.04$"),
                      (133, sig_pca, "Sigmoid kernel, $\gamma=10^{-3}, r=1$"))

    best_kernel = ""
    best_kernel_err = math.inf
    for subplot, pca, title in kernel_options:
        pca.fit(X_train)
        X_test_reduced = pca.transform(X_test)
        X_test_preimage = pca.inverse_transform(X_test_reduced)
        err = mean_squared_error(X_test, X_test_preimage)
        print("Kernel PCA (", title, ") MSE reconstruction loss:", err)
        if err < best_kernel_err:
            best_kernel = title
            best_kernel_err = err
            print("- New best kernel")

    print("\n\nBest MSE reconstruction error:", best_kernel_err)
    print("- Kernel:", best_kernel)

    return best_kernel, best_kernel_err


def plot3clusters(X, title, vtitle, target_names):
    colors = ['#A43F98', '#5358E0', '#DE0202']
    s = 50
    alpha = 0.7

    plt.figure(figsize=(9, 7))
    plt.grid(True)
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(X[y == i, 0], X[y == i, 1], color=color,
                    alpha=alpha, s=s, label=target_name)

    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title(title, fontsize=16, fontweight='bold')

    plt.text(0.5, -0.1, 'Principal Component 1', ha='center',
             fontsize=12, fontweight='bold', transform=plt.gca().transAxes)
    plt.text(-0.1, 0.5, 'Principal Component 2', va='center', rotation='vertical',
             fontsize=12, fontweight='bold', transform=plt.gca().transAxes)

    plt.show()

# +


def autoencoder_exp(X_train, X_test, k):
    input_dim = X_train.shape[1]  # input shape
    output_dim = X_train.shape[1]
    encoding_dim = k  # encoding dimension - #neurons for the dense layers
    optimizer = 'adam'
    loss = 'mse'

    input_layer = tf.keras.Input(
        shape=(input_dim,), name='input')  # input layer
    encoding_layer = tf.keras.layers.Dense(
        encoding_dim, activation='relu', name='encoding')(input_layer)  # encoding layer
    decoding_layer = tf.keras.layers.Dense(
        output_dim, activation='sigmoid', name='decoding')(encoding_layer)  # decoding layer

    input_layer = tf.keras.Input(
        shape=(input_dim,), name='input')  # input layer
    encoding_layer = tf.keras.layers.Dense(
        encoding_dim, name='encoding')(input_layer)  # encoding layer
    decoding_layer = tf.keras.layers.Dense(
        output_dim, name='decoding')(encoding_layer)  # decoding layer

    autoencoder = tf.keras.Model(input_layer, decoding_layer)
    autoencoder.compile(optimizer=optimizer, loss=loss)
    autoencoder.summary()

    # Set other parameters
    epochs = 50
    batch_size = 16
    shuffle = True
    validation_split = 0.1
    verbose = 0

    history = autoencoder.fit(X_train, X_train,
                              epochs=epochs,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              validation_split=validation_split,
                              verbose=verbose)

    # Plot the loss
    plt.plot(history.history['loss'], color='#FF7E79', linewidth=3, alpha=0.5)
    plt.plot(history.history['val_loss'],
             color='#007D66', linewidth=3, alpha=0.4)
    plt.title('Model train vs Validation loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='best')
    plt.grid(True)
    plt.show()

    print("Training loss:", history.history['loss'][-1])
    print("Validation loss:", history.history['val_loss'][-1])

    # Compute MSE on train and test data
    autoencoder_err_train = mean_squared_error(X_train, autoencoder(X_train))
    autoencoder_err_test = mean_squared_error(X_test, autoencoder(X_test))
    print("MSE on training data:", autoencoder_err_train)
    print("MSE on test data:", autoencoder_err_test)

    if k == 2:  # can only plot if there are 2 dims
        encoder = tf.keras.Model(input_layer, encoding_layer)
        encoded_data = encoder.predict(X_train)

#         target_names = iris.target_names

#         plot3clusters(encoded_data, 'Encoded data latent-space', 'dimension ', target_names);

    return autoencoder_err_train, autoencoder_err_test
# -


def nmf_exp(X_train, X_test, k):
    print("\n\nBest NMF:")

    # Perform NMF
    model = NMF(n_components=k, init='random', random_state=109)
    W_train = model.fit_transform(X_train)
    H_train = model.components_

    # Print the basis vectors and the coefficients
    print("Basis vectors:\n", H_train)
    print("Coefficients:\n", W_train)

    # Reconstruct train data and compute MSE
    X_train_reconstructed = np.dot(W_train, H_train)
    train_mse = np.mean((X_train - X_train_reconstructed) ** 2)
    print('\n\nBest MSE reconstruction error on train data:', train_mse)

    # Reconstruct test data and compute MSE
    W_test = model.transform(X_test)
    H_test = model.components_
    X_test_reconstructed = np.dot(W_test, H_test)
    test_mse = np.mean((X_test - X_test_reconstructed) ** 2)
    print('Best MSE reconstruction error on test data:', test_mse)

    return train_mse, test_mse


def nmf_exp_faces(X_train, X_test, k, h, w):
    print("\n\nBest NMF:")

    # Perform NMF
    model = NMF(n_components=k, init='random', random_state=109)
    W_train = model.fit_transform(X_train)
    H_train = model.components_
    eigenfaces_nmf = model.components_.reshape((k, h, w))

    # Print the basis vectors and the coefficients
    print("Basis vectors:\n", H_train)
    print("Coefficients:\n", W_train)

    # Reconstruct train data and compute MSE
    X_train_reconstructed = np.dot(W_train, H_train)
    train_mse = np.mean((X_train - X_train_reconstructed) ** 2)
    print('\n\nBest MSE reconstruction error on train data:', train_mse)

    # Reconstruct test data and compute MSE
    W_test = model.transform(X_test)
    H_test = model.components_
    X_test_reconstructed = np.dot(W_test, H_test)
    test_mse = np.mean((X_test - X_test_reconstructed) ** 2)
    print('Best MSE reconstruction error on test data:', test_mse)

    return train_mse, test_mse, eigenfaces_nmf
