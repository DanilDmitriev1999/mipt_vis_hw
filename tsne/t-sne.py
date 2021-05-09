import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import warnings

warnings.filterwarnings('ignore')


class TSNE:
    def __init__(self, max_iter, lr=500, n_components=2, labels=None, print_log=True,):
        self.max_iter = max_iter
        self.lr = lr
        self.n_components = n_components
        self.labels = labels
        self.print_log = print_log

    def binary_search(self, dist, target_entropy, perplexity_tries=50):
        precision_min = 0
        precision_max = 1.0e15
        precision = 1.0e5

        for _ in range(perplexity_tries):
            denom = np.sum(np.exp(-dist[dist > 0.0] / precision))
            beta = np.exp(-dist / precision) / denom

            g_beta = beta[beta > 0.0]
            entropy = -np.sum(g_beta * np.log2(g_beta))

            error = entropy - target_entropy

            if error > 0:
                precision_max = precision
                precision = (precision + precision_min) / 2.0
            else:
                precision_min = precision
                precision = (precision + precision_max) / 2.0

            if np.abs(error) < target_entropy:
                break

        return beta

    def plot_tsne(self, Y):
        plt.figure(figsize=(13,10))
        plt.scatter(Y[:, 0], Y[:, 1], c=self.labels, cmap="jet")
        plt.axis('off')
        plt.colorbar()
        plt.show()

    def compute_p(self, X):
        n, d = X.shape
        P = np.zeros((n, n), dtype=np.float32)
        target_entropy = np.log(30)
        dist = self.l2_dist(X)

        for i in range(n):
            P[i, :] = self.binary_search(dist[i], target_entropy)

        np.fill_diagonal(P, 1e-12)
        P = P.clip(min=1e-100)
        P = (P + P.T) / (2 * n)

        return P

    def transform(self, X, plot_result=True):
        n, d = X.shape
        min_gain = 0.01

        Y = np.random.normal(0., 0.0001, [n, self.n_components])
        dY = np.zeros((n, self.n_components))
        iY = np.zeros((n, self.n_components))
        grad = np.ones((n, self.n_components))

        # P-values
        P = self.compute_p(X)

        # compute grad
        for idx_iter in range(self.max_iter):
            sum_Y = np.sum(np.square(Y), 1)
            num = -2. * np.dot(Y, Y.T)
            num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
            num[range(n), range(n)] = 0.
            output_prob = num / np.sum(num)
            output_prob = np.maximum(output_prob, 1e-12)

            momentum = 0.5 if idx_iter < 20 else 0.8

            PQ = P - output_prob
            for i in range(n):
                dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (self.n_components, 1)).T * (Y[i, :] - Y), 0)

            grad = (grad + 0.2) * ((dY > 0.) != (iY > 0.)) + (grad * 0.8) * ((dY > 0.) == (iY > 0.))
            grad[grad < min_gain] = min_gain
            iY = momentum * iY - self.lr * (grad * dY)
            Y = Y + iY
            Y = Y - np.tile(np.mean(Y, 0), (n, 1))

            if idx_iter % 50 == 0 and self.print_log:
                error = np.sum(P * np.log(P / output_prob))
                print("Iteration %s, error %s" % (idx_iter, error))

        if plot_result and self.labels is not None:
            self.plot_tsne(Y)

        return Y

    @staticmethod
    def l2_dist(X):
        sum_X = np.sum(np.square(X), 1)
        return (-2 * np.dot(X, X.T) + sum_X).T + sum_X


if __name__ == '__main__':
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target

    ts = TSNE(800, 20, 2, y)
    Y = ts.transform(X, True)
