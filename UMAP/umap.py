import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import datasets
import warnings

warnings.filterwarnings('ignore')


class UMAP:
    def __init__(self, n_iter=150, lr=0.5, labels=None):
        self.n_iter = n_iter
        self.lr = lr
        self.labels = labels

        self.a = 1.5
        self.b = 0.88

    @staticmethod
    def prob_high_dim(sigma, dist_row, dist, rho):
        d = dist[dist_row] - rho[dist_row]
        d[d < 0] = 0
        return np.exp(- d / sigma)

    @staticmethod
    def sigma_binary_search(k_of_sigma, fixed_k):
        sigma_lower = 0
        sigma_upper = 1000
        for i in range(20):
            approx_sigma = (sigma_lower + sigma_upper) / 2
            if k_of_sigma(approx_sigma) < fixed_k:
                sigma_lower = approx_sigma
            else:
                sigma_upper = approx_sigma
            if np.abs(fixed_k - k_of_sigma(approx_sigma)) <= 1e-5:
                break
        return approx_sigma

    def plot_tsne(self, Y):
        plt.figure(figsize=(13, 10))
        plt.scatter(Y[:, 0], Y[:, 1], c=self.labels, cmap="jet")
        plt.axis('off')
        plt.colorbar()
        plt.show()

    def compute_grad(self, P, Y):
        y = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)
        dist = np.power(1 + self.a * np.square(euclidean_distances(Y, Y)) ** self.b, -1)
        PQ = np.dot(1 - P, np.power(0.001 + np.square(euclidean_distances(Y, Y)), -1))
        np.fill_diagonal(PQ, 0)
        PQ /= np.sum(PQ, axis=1)
        fact = np.expand_dims(self.a * P * (1e-8 + np.square(euclidean_distances(Y, Y))) ** (self.b - 1) - PQ, 2)
        return 2 * self.b * np.sum(fact * y * np.expand_dims(dist, 2), axis=1)

    def compute_p(self, X):
        n = X.shape[0]
        dist = np.square(euclidean_distances(X, X))
        rho = [sorted(dist[i])[1] for i in range(dist.shape[0])]
        prob = np.zeros((n, n))
        sigma_array = []
        f = lambda prob: np.power(2, np.sum(prob))
        for dist_row in range(n):
            func = lambda sigma: f(self.prob_high_dim(sigma, dist_row, dist, rho))
            binary_search_result = self.sigma_binary_search(func, 10)
            prob[dist_row] = self.prob_high_dim(binary_search_result, dist_row, dist, rho)
            sigma_array.append(binary_search_result)

        print("\nMean sigma = " + str(np.mean(sigma_array)))

        P = (prob + np.transpose(prob)) / 2
        return P

    def transform(self, X, n_dims=2, plot_result=True):
        n = X.shape[0]
        X = np.log(X + 1)
        P = self.compute_p(X)
        y = np.random.normal(loc=0, scale=1, size=(n, n_dims))

        for i in range(self.n_iter):
            y = y - self.lr * self.compute_grad(P, y)

        if plot_result and self.labels is not None:
            self.plot_tsne(y)

        return None


if __name__ == '__main__':
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target

    u = UMAP(labels=y)
    u.transform(X)
