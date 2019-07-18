import numpy as np
from emalgorithm import make_multivariate_random, EMAlgorithm

if __name__ == '__main__':

    K = 3
    N = 1000
    mu = [[-4, 0.5],
          [3, 0.72],
          [7, 5]]
    sigma = [[[3, 2],
              [2, 5]],
             [[2, -2],
              [-2, 4]],
             [[2, 1],
              [1, 4]]]
    pi = [0.3, 0.2, 0.5]

    X, X_plot = make_multivariate_random(K, N, mu, sigma, pi)

    em = EMAlgorithm(K, X, X_plot)
    em.animate()
