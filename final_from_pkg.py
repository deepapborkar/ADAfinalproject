'''This file uses built in packages for SVD and NMF for the netflix data'''
import sys
import numpy as np
import pandas as pd
from scipy import sparse
from numpy.linalg import matrix_rank
from scipy.linalg import svd
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt

# FUNCTIONS

def rmse_function(actual, estimate):
    '''
    This function calculates the root mean square error between 
    two matrices (called actual and estimate)
    
    Parameters:
    actual (ndarray): a 2-D array with the actual known ratings where
                        rows represent users and columns represent movies
                        (zero values are in the place of unknown ratings)
    estimate (ndarray): a 2-D array with the estimated ratings for the
                        same users and movies
    Returns:
    rmse_val (float): root mean square error
    '''
    rmse_val = 0
    for row, col in np.ndindex(actual.shape):
        # only calculate RMSE for ratings that exist
        if actual[row, col] != 0:
            rmse_val += (actual[row, col] - estimate[row, col])**2
    rmse_val = np.sqrt(rmse_val)
    return rmse_val

def matrix_factorization(sparse_matrix):
    '''
    This function runs the SVD and NMF packages on the sparse matrix

    Parameters:
        sparse_matrix (csr_matrix): input matrix with users and movies as indices and 
        known ratings as values
    Returns:
        rmse_svd (float): root mean square error for the estimate vs input matrix using SVD method
        rmse_svd (float): root mean square error for the estimate vs input matrix using NMF method
    '''
    # convert to a numpy array
    sparse_matrix_arr = sparse.csr_matrix.toarray(sparse_matrix)

    # parameters for SVD and NMF
    num_components = matrix_rank(sparse_matrix_arr) # num_components is usually the rank of the matrix

    # SVD Method from sklearn.decomposition package
    truncated_svd = TruncatedSVD(num_components)
    reduced_matrix = truncated_svd.fit_transform(sparse_matrix)
    estimate_svd = truncated_svd.inverse_transform(reduced_matrix)
    
    # calculate root mean square error using rmse function
    rmse_svd = rmse_function(sparse_matrix_arr, estimate_svd)
    #print(estimate_svd)
    #print('rmse for svd')
    #print(rmse_svd)

    # NMF Method from sklearn.decomposition package
    nmf_method = NMF(num_components, max_iter = 1000, beta_loss = 'frobenius', init = 'nndsvd')
    reduced_matrix = nmf_method.fit_transform(sparse_matrix)
    estimate_nmf = nmf_method.inverse_transform(reduced_matrix)
    
    # calculate root mean square error using rmse function
    rmse_nmf = rmse_function(sparse_matrix_arr, estimate_nmf)
    #print(estimate_nmf)
    #print('rmse for nmf')
    #print(rmse_nmf)

    return rmse_svd, rmse_nmf

# MAIN PROGRAM

# read in the csv data file from user input (i.e. 'data/data.csv')
data_dir = sys.argv[1]

# read in csv to a dataframe
df = pd.read_csv(data_dir, sep=',', names=['movie_id','user_id','rating','date'])

# subtract 1 from the user_id and movie_id so the indices start with 0
df['movie_id'] = df['movie_id'] - 1
df['user_id'] = df['user_id'] - 1

# section for comparing SVD and NMF
samples = [i for i in range(100, 5000, 100)]
rmse_svd_lst = []
rmse_nmf_lst = []

for sample_size in samples:

    # select varying sample size of user_ids and movie_ids (memory issues when trying to use all of the data)
    df = df[df['user_id'] < sample_size]
    df = df[df['movie_id'] < sample_size]

    # convert df to a sparse matrix (utility matrix)
    sparse_matrix = sparse.csr_matrix((df.rating.values, (df.user_id.values, df.movie_id.values)),)

    # calculate rmse for SVD and NMF
    rmse_svd, rmse_nmf = matrix_factorization(sparse_matrix)

    rmse_svd_lst.append(rmse_svd)
    rmse_nmf_lst.append(rmse_nmf)

# plot the rmse error vs sample size

plt.title("Comparing MF Methods by RMSE")
plt.xlabel("Sample Size")
plt.ylabel("RMSE")
plt.plot(samples, rmse_svd_lst, label = 'SVD')
plt.plot(samples, rmse_nmf_lst, label = 'NMF')
plt.legend()
plt.savefig('plot_pkg.png')


# section for testing purposes

# use SVD from scipy.linalg package to get U, sigma, and V
#U, sigma, V = svd(sparse_matrix_arr, full_matrices=False)

#print('U shape')
#print(U.shape)
#print('sigma shape')
#print(sigma.shape)
#print('V shape')
#print(V.shape)

#estimate = np.dot(U.dot(np.diag(sigma)), V)

#rmse = rmse_function(sparse_matrix_arr, estimate)
#print('rmse from scipy package for svd')
#print(rmse)


