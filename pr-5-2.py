import numpy as np
import seaborn as sns
from scipy.stats import norm
from scipy import stats as st
import math

import matplotlib
from matplotlib import font_manager
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
import matplotlib.animation as ani

min_x, min_y = -10, -10
max_x, max_y = 10, 10

def calc_prob_gmm(data, mu, sigma, pi, K):
    return [[pi[k]*st.multivariate_normal.pdf(d, mu[k], sigma[k]) for k in range(K)] for d in data]

def print_gmm_contour(mu, sigma, pi, K):
    # display predicted scores by the model as a contour plot

    X, Y = np.meshgrid(np.linspace(min_x, max_x), np.linspace(min_y, max_y))
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = np.sum(np.asanyarray(calc_prob_gmm(XX, mu, sigma, pi, K)), axis=1)
    Z = Z.reshape(X.shape)
    CS = plt.contour(X, Y, Z, alpha=0.2, zorder=-100)

def mixture_plot(K, n, mu, sigma, pi):
    # Probability Density Function
    xx = np.linspace(-10, 10, n)
    pdfs = np.zeros((n, K))
    for k in range(K):
        pdfs[:, k] = pi[k] * st.norm.pdf(xx, loc=mu[k], scale=sigma[k])

    plt.figure(figsize=(14, 6))
    for k in range(K):
        plt.plot(xx, pdfs[:, k])
    plt.title("pdfs")
    plt.show()

    plt.figure(figsize=(14, 6))
    plt.stackplot(xx, pdfs[:, 0], pdfs[:, 1], pdfs[:, 2])
    plt.title("stacked")
    plt.show()

def make_random(K, N, mu, sigma, pi, plot=False):
    """
    make one dimention gauss random number
    K     : num of mixture element
    N     : size of random number
    mu    : list of mu
    sigma : list of sigma
    pi    : list of muxture coefficient
    """
    X = None
    X_plot = []
    for k in range(K):
        X_k = np.random.normal(loc=mu[k], scale=sigma[k], size=int(N*pi[k]))
        X_plot.append(X_k)
        if X is not None:
            X = np.concatenate([X, X_k])
        else:
            X = X_k
    if plot == True:
        plt.hist(X_plot)
        plt.show()

    return X.reshape(N, 1), X_plot

def make_multivariate_random(K, N, mu, sigma, pi, plot=False):
    """
    make one dimention gauss random number
    K     : num of mixture element
    N     : size of random number
    mu    : list of mu
    sigma : list of sigma
    pi    : list of muxture coefficient
    """
    X = None
    X_plot = []
    for k in range(K):
        X_k = np.random.multivariate_normal(mu[k], sigma[k], size=int(N*pi[k]))
        X_plot.append(X_k)
        if X is not None:
            X = np.concatenate([X, X_k])
        else:
            X = X_k
    if plot == True:
        for X_p in X_plot:
            plt.scatter(X_p[:, 0], X_p[:, 1])
        plt.show()

    return X, X_plot



class EMAlgorithm:
    """
    num pf mixture element is hyper parameter
    """
    def __init__(self, K, X, X_plot):
        """
        set data and shape of data
        """
        self.K = K              # num of mixture element
        self.X = X              # train data (np.ndarray)
        self.N = X.shape[0]     # num of train data
        self.dim = X.shape[1]   # dimention of variables

        self.X_plot = X_plot
        """
        initialize mu, sigma and pi using uniform distribution
        """
        # initialize avarage vector (size = (mixture_num, data_dim))
        self.mu = np.random.uniform(low=X.min(),
                                    high=X.max(),
                                    size=(self.K, self.dim))
        # initialize cov vector (size = (mixture_num, data_dim, data_dim))
        # self.sigma = np.random.uniform(low=0.5*np.var(self.X),
        #                                high=1.5*np.var(self.X),
        #                                size=(self.K, self.dim, self.dim))
        self.sigma = np.array([np.identity(self.dim) for k in range(K)]).reshape((self.K, self.dim, self.dim))
        # initialize mixture coefficient (size = (self.K,))
        self.pi = np.ones(self.K) / self.K
        """
        initialize for utility
        """
        # make ndarray for responsibility : size = (data_num. data_dim)
        self.gamma = np.zeros((self.N, self.K))

        # print("K = {}, N = {}, dim = {}".format(self.K, self.N, self.dim))
        # print(self.mu)
        # print(self.sigma)

    def estep(self, X):
        """
        compute on E step (renew responsibility)
        """
        for n in range(self.N):
            down = 0
            for k in range(self.K):
                down += self.pi[k]*self.gauss_pdf(X[n], self.mu[k], self.sigma[k])

            for k in range(self.K):
                up = self.pi[k]*self.gauss_pdf(X[n], self.mu[k], self.sigma[k])
                self.gamma[n, k] = up / down

    def mstep(self, X):
        """
        compute on M step (renew parameter)
        """
        # renew pi
        # self.pi = np.sum(self.gamma, axis=0)
        for k in range(self.K):

            # renew pi
            self.pi[k] = np.sum(self.gamma[:,k]) / self.N

            # renew mu
            up_mu = 0
            for n in range(self.N):
                up_mu += self.gamma[n,k] * X[n]

            self.mu[k] = up_mu / np.sum(self.gamma[:,k])

            # renew sigma
            up_sigma = 0
            for n in range(self.N):
                vec = X[n]-self.mu[k]
                vec = vec[:, None] # expand dimention
                up_sigma += self.gamma[n,k] * np.dot(vec, vec.T)

            self.sigma[k] = up_sigma / np.sum(self.gamma[:,k])

    def fit(self, X_plot):
        """
        execute optimization on EM algorithm
        """

        while True:
            # current log likelihood function
            L_previous = 0
            for n in range(self.N):
                q_previous = 0
                for k in range(self.K):
                    q_previous += self.pi[k]*self.gauss_pdf(self.X[n], self.mu[k], self.sigma[k])
                L_previous += np.log(q_previous)

            # fitting
            self.estep(self.X)
            self.mstep(self.X)

            # Convergence judgment
            L_now = 0
            for n in range(self.N):
                q_now = 0
                for k in range(self.K):
                    q_now += self.pi[k]*self.gauss_pdf(self.X[n], self.mu[k], self.sigma[k])
                L_now += np.log(q_now)


            print("Likelihood : {},  {}".format(L_previous, L_now))

            # visualize
            plt.cla()
            color1 = ['blue', 'red', 'green', 'yellow']
            color2 = ['salmon', 'orangered', 'gold', 'coral']
            i = 0
            for X in self.X_plot:
                im = plt.scatter(X[:, 0], X[:, 1], marker='+', s=5, color=color1[i])
                i += 1
            for k in range(K):
                im = plt.scatter(self.mu[k, 0], self.mu[k, 1], marker='s', s=20, color=color2[k])

            if np.allclose(L_previous, L_now):
                break
        print("finished fitting")

    def anim_plot(self, X_plot):

        # current log likelihood function
        L_previous = 0
        for n in range(self.N):
            q_previous = 0
            for k in range(self.K):
                q_previous += self.pi[k]*self.gauss_pdf(self.X[n], self.mu[k], self.sigma[k])
            L_previous += np.log(q_previous)

        # fitting
        self.estep(self.X)
        self.mstep(self.X)

        # Convergence judgment
        L_now = 0
        for n in range(self.N):
            q_now = 0
            for k in range(self.K):
                q_now += self.pi[k]*self.gauss_pdf(self.X[n], self.mu[k], self.sigma[k])
            L_now += np.log(q_now)


        print("Likelihood : {},  {}".format(L_previous, L_now))

        # visualize
        plt.cla()
        color1 = ['blue', 'red', 'green', 'yellow']
        color2 = ['salmon', 'orangered', 'gold', 'coral']
        i = 0
        for X in self.X_plot:
            im = plt.scatter(X[:, 0], X[:, 1], marker='+', s=5, color=color1[i])
            i += 1
        for k in range(K):
            im = plt.scatter(self.mu[k, 0], self.mu[k, 1], marker='s', s=20, color=color2[k])
        print_gmm_contour(self.mu, self.sigma, self.pi, self.K)

    def animate(self):

        fig = plt.figure(figsize=(12,5))
        anim = ani.FuncAnimation(fig, self.anim_plot, interval=100, frames=35)
        anim.save('gmm_em.gif', writer='imagemagick', fps=3, dpi=128)


    def gauss_pdf(self, x, mu, sigma):
        """
        return probability density function value on x
        """
        if x.shape[0] == 1:
            if sigma <= 0:
                sigma += 1e-8
            pdf = st.norm.pdf(x=x, loc=mu, scale=sigma)
        else:
            pdf = st.multivariate_normal.pdf(x=x, mean=mu, cov=sigma)
        return pdf


if __name__ == '__main__':
    # K = 3 # num of mixture
    # N = 1000
    # mu = [-3, 0, 3]
    # sigma = [0.7, 1, 1.4]
    # pi = [0.3, 0.2, 0.5]
    #X, X_plot = make_random(K, N, mu, sigma, pi)

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
