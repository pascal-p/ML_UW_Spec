from scipy.sparse import csr_matrix
from scipy.sparse import spdiags
from scipy.stats import multivariate_normal
import turicreate
import numpy as np
import sys
import time
from copy import deepcopy
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

def sframe_to_scipy(x, column_name):
    """
    Convert a dictionary column of an SFrame into a sparse matrix format where
    each (row_id, column_id, value) triple corresponds to the value of
    x[row_id][column_id], where column_id is a key in the dictionary.

    Example
    >>> sparse_matrix, map_key_to_index = sframe_to_scipy(sframe, column_name)
    """
    assert type(x[column_name][0]) == dict, 'The chosen column must be dict type, representing sparse data.'

    ## Stack will transform x to have a row for each unique (row, key) pair.
    x = x.stack(column_name, ['feature', 'value'])

    ## Map feature words to integers and conversely (rev_mapping)
    mapping = {word: i for (i, word) in enumerate(sorted(x['feature'].unique()))}
    rev_mapping = {i: word for (i, word) in enumerate(sorted(x['feature'].unique()))}
    x['feature_id'] = x['feature'].apply(lambda x: mapping[x])

    ## Create numpy arrays that contain the data for the sparse matrix.
    row_id = np.array(x['id'])
    col_id = np.array(x['feature_id'])
    data = np.array(x['value'])

    width = x['id'].max() + 1
    height = x['feature_id'].max() + 1

    ## Create a sparse matrix.
    mat = csr_matrix((data, (row_id, col_id)), shape=(width, height))
    return (mat, mapping, rev_mapping)

def diag(array):
    n = len(array)
    return spdiags(array, 0, n, n)

def logpdf_diagonal_gaussian(x, mean, cov):
    """
    Compute logpdf of a multivariate Gaussian distribution with diagonal covariance at a given point x.
    A multivariate Gaussian distribution with a diagonal covariance is equivalent
    to a collection of independent Gaussian random variables.

    x should be a sparse matrix. The logpdf will be computed for each row of x.
    mean and cov should be given as 1D numpy arrays
    mean[i] : mean of i-th variable
    cov[i] : variance of i-th variable
    """

    n = x.shape[0]
    dim = x.shape[1]
    assert(dim == len(mean) and dim == len(cov))


    two_sigma = 2 * np.sqrt(cov)

    ## multiply each i-th column of x by (1 / (2 * sigma_i)), where sigma_i is sqrt of variance of i-th variable.
    scaled_x = x.dot(diag(1. / two_sigma))

    ## multiply each i-th entry of mean by (1 / (2 * sigma_i))
    scaled_mean = mean / two_sigma

    ## sum of pairwise squared Eulidean distances gives SUM[(x_i - mean_i)^2/(2*sigma_i^2)]
    return -np.sum(np.log(np.sqrt(2. * np.pi * cov))) - pairwise_distances(scaled_x, [scaled_mean],
                                                                           'euclidean').flatten() ** 2

def log_sum_exp(x, axis):
    """
    Compute the log of a sum of exponentials
    """
    x_max = np.max(x, axis=axis)
    if axis == 1:
        return x_max + np.log(np.sum(np.exp(x - x_max[:, np.newaxis]), axis=1))
    else:
        return x_max + np.log(np.sum(np.exp(x - x_max), axis=0) )

def EM_for_high_dimension(data, means, covs, weights, cov_smoothing=1e-5, maxiter=int(1e3), thresh=1e-4, verbose=False):
    # cov_smoothing: specifies the default variance assigned to absent features in a cluster.
    #                If we were to assign zero variances to absent features, we would be overconfient,
    #                as we hastily conclude that those featurese would NEVER appear in the cluster.
    #                We'd like to leave a little bit of possibility for absent features to show up later.
    n, dim = data.shape[0], data.shape[1]
    mu, Sigma = deepcopy(means), deepcopy(covs)
    K = len(mu)
    weights = np.array(weights)
    ll = None
    ll_trace = []

    for i in range(maxiter):
        ## E-step: compute responsibilities
        logresp = np.zeros((n, K))
        for k in range(K):
            logresp[:,k] = np.log(weights[k]) + logpdf_diagonal_gaussian(data, mu[k], Sigma[k])

        ll_new = np.sum(log_sum_exp(logresp, axis=1))
        if verbose: print(ll_new)
        sys.stdout.flush()

        logresp -= np.vstack(log_sum_exp(logresp, axis=1))
        resp = np.exp(logresp)
        counts = np.sum(resp, axis=0)

        ## M-step: update weights, means, covariances
        weights = counts / np.sum(counts)
        for k in range(K):
            mu[k] = (diag(resp[:,k]).dot(data)).sum(axis=0)/counts[k]
            mu[k] = mu[k].A1
            Sigma[k] = diag(resp[:, k]).dot(data.multiply(data) - 2. * data.dot(diag(mu[k]))).sum(axis=0) + (mu[k] ** 2) * counts[k]
            Sigma[k] = Sigma[k].A1 / counts[k] + cov_smoothing * np.ones(dim)

        ## check for convergence in log-likelihood
        ll_trace.append(ll_new)
        if ll is not None and (ll_new - ll) < thresh and ll_new > -np.inf:
            ll = ll_new
            break
        ll = ll_new
    ##
    return {'weights':weights, 'means':mu, 'covs':Sigma, 'loglik':ll_trace, 'resp':resp}
