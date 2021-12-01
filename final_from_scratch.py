'''This file has the function for SVD and NMF from scratch for the test and sample data'''
import sys
import numpy as np
import pandas as pd
from scipy import sparse
from numpy.linalg import matrix_rank
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
    count = 0
    for row, col in np.ndindex(actual.shape):
        # only calculate RMSE for ratings that exist
        if actual[row, col] != 0:
            rmse_val += (actual[row, col] - estimate[row, col])**2
            count += 1
    rmse_val = np.sqrt(rmse_val/count)
    return rmse_val

def sgd_svd(arr, n_components, lr, epochs):
    '''
    This function calculates the users and items matrix factorization
    of the input arr and also the root mean square error of each epoch
    using the SVD matrix factorization method

    Parameters:
    arr (ndarray): a 2-D array with the actual known ratings where
                        rows represent users and columns represent movies
                        (zero values are in the place of unknown ratings)
    n_components (int): the number of features (usually the rank of arr)
    lr (float): learning rate
    epochs (int): number of runs in order to find an estimate of the ratings

    Returns:
    u (ndarray): array that considers the latent user feature set
    i (ndarray): array that considers the latent movie feature set
    rmse_arr (ndarray): rmse values for each epoch run
    '''
    m = arr.shape[0]
    n = arr.shape[1]
    r = n_components

    # randomly initialize the matrices u and i
    u = np.random.uniform(0, 1, (m, r))
    i = np.random.uniform(0, 1, (n, r))
    # initialize the matrices u and i
    #u = np.zeros((m, r))
    #i = np.zeros((n, r))

    # keeps track of the rmse
    rmse_arr = np.zeros(epochs)

    # go through epochs
    for e in range(epochs):
        #print('epoch: ', e)
        # go through all the ratings in arr
        for row, col in np.ndindex(arr.shape):
            # only look at the ratings that exist (not equal to zero)
            if arr[row, col] != 0:
                rating = arr[row, col]
                # partial derivatives
                u[row] += 2 * lr * (rating - u[row].dot(i[col])) * i[col]
                i[col] += 2 * lr * (rating - u[row].dot(i[col])) * u[row]
        rmse_arr[e] = rmse_function(arr, u.dot(i.T))

    return u, i.T, rmse_arr

def sgd_nmf(arr, n_components, lr, epochs):
    '''
    This function calculates the users and items matrix factorization
    of the input arr and also the root mean square error of each epoch
    using the NMF matrix factorization method

    Parameters:
    arr (ndarray): a 2-D array with the actual known ratings where
                        rows represent users and columns represent movies
                        (zero values are in the place of unknown ratings)
    n_components (int): the number of features (usually the rank of arr)
    lr (float): learning rate
    epochs (int): number of runs in order to find an estimate of the ratings

    Returns:
    u (ndarray): array that considers the latent user feature set
    i (ndarray): array that considers the latent movie feature set
    rmse_arr (ndarray): rmse values for each epoch run
    '''
    m = arr.shape[0]
    n = arr.shape[1]
    r = n_components

    # randomly initialize the matrices u and i
    u = np.random.uniform(0, 1, (m, r))
    i = np.random.uniform(0, 1, (n, r))
    # initialize the matrices u and i
    #u = np.zeros((m, r))
    #i = np.zeros((n, r))

    # keeps track of the rmse
    rmse_arr = np.zeros(epochs)

    # go through epochs
    for e in range(epochs):
        #print('epoch:', e)
        # go through all the ratings in arr
        for row, col in np.ndindex(arr.shape):
            # only look at the ratings that exist (not equal to zero)
            if arr[row, col] != 0:
                rating = arr[row, col]
                # partial derivatives
                u[row] += 2 * lr * (rating - u[row].dot(i[col])) * i[col]
                i[col] += 2 * lr * (rating - u[row].dot(i[col])) * u[row]
                # put constraint that all values must be nonnegative
                u[row][u[row] < 0] = 0
                i[col][i[col] < 0] = 0
        rmse_arr[e] = rmse_function(arr, u.dot(i.T))
    return u, i.T, rmse_arr

def svd_function(arr, n_components = 2, lr = 0.01, epochs = 1000):
    '''
    This function calculates the estimate of the ratings based on the SVD method

    Parameters:
    arr (ndarray): a 2-D array with the actual known ratings where
                        rows represent users and columns represent movies
                        (zero values are in the place of unknown ratings)
    n_components (int): the number of features (usually the rank of arr)
    lr (float): learning rate
    epochs (int): number of runs in order to find an estimate of the ratings

    Returns:
    estimate (ndarray): estimated ratings that the users gave the movies
    rmse_arr (ndarray): rmse values for each epoch run
    '''
    # use stochastic gradient descent to get user and item matrices (arr decomposition)
    users, items, rmse_arr = sgd_svd(arr, n_components, lr, epochs)
    #print('users')
    #print(users)
    #print('items')
    #print(items)
    # calculate the new rating estimates
    estimate = users.dot(items)
    return estimate, rmse_arr

def nmf_function(arr, n_components = 2, lr = 0.01, epochs = 1000):
    '''
    This function calculates the estimate of the ratings based on the NMF method

    Parameters:
    arr (ndarray): a 2-D array with the actual known ratings where
                        rows represent users and columns represent movies
                        (zero values are in the place of unknown ratings)
    n_components (int): the number of features (usually the rank of arr)
    lr (float): learning rate
    epochs (int): number of runs in order to find an estimate of the ratings

    Returns:
    estimate (ndarray): estimated ratings that the users gave the movies
    rmse_arr (ndarray): rmse values for each epoch run
    '''
    # use stochastic gradient descent to get user and item matrices (arr decomposition)
    users, items, rmse_arr = sgd_nmf(arr, n_components, lr, epochs)
    #print('users')
    #print(users)
    #print('items')
    #print(items)
    # calculate the new rating estimates
    estimate = users.dot(items)
    return estimate, rmse_arr

# MAIN PROGRAM

# read in the csv data file from user input (i.e. 'data/sample_data.csv')
data_dir = sys.argv[1]

# read the data into a dataframe

df = pd.read_csv(data_dir, sep=',', names=['movie_id','user_id','rating','date'])

# subtract 1 from the user_id and movie_id so the indices start with 0
df['movie_id'] = df['movie_id'] - 1
df['user_id'] = df['user_id'] - 1

# only select the first 1000 user_ids and movie_ids
#df = df[df['user_id'] < 1000]
#df = df[df['movie_id'] < 1000]
#print(df.head())
#print(df.describe())

# convert df to a sparse matrix (utility matrix)
sparse_matrix = sparse.csr_matrix((df.rating.values, (df.user_id.values, df.movie_id.values)),)
# convert to a numpy array
sparse_matrix = sparse.csr_matrix.toarray(sparse_matrix)

#print(sparse_matrix.shape)

#print(sparse_matrix)

# parameters for SVD and NMF
num_components = matrix_rank(sparse_matrix) # num_components is usually the rank of the matrix
#print(num_components)
learning_rate = 0.001
num_epochs = 3000
# create a list of epochs for plotting purposes
epochs_lst = [i for i in range(num_epochs)]

rating_estimates_svd, rmse_svd = svd_function(sparse_matrix, num_components, learning_rate, num_epochs)
rating_estimates_nmf, rmse_nmf = nmf_function(sparse_matrix, num_components, learning_rate, num_epochs)

#print(rating_estimates_svd)
#print(rmse_svd)
#print(rating_estimates_nmf)
#print(rmse_nmf)

# plot the rmse error vs epochs

plt.title("Comparing MF Methods by RMSE")
plt.xlabel("Epoch #")
plt.ylabel("RMSE")
plt.plot(epochs_lst, rmse_svd, label = 'SVD')
plt.plot(epochs_lst, rmse_nmf, label = 'NMF')
plt.legend()
plt.savefig('plot.png')

print('RMSE for SVD')
print(rmse_svd[-1])
print('RMSE for NMF')
print(rmse_nmf[-1])
