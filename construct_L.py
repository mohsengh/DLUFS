import numpy as np


def construct_L(X):
    n = len(X[:, 0])
    Dist = np.zeros([n, n])

    S_temp = np.zeros([n, n])

    for i in range(0, n):
        for j in range(0, n):
            Dist[i, j] = np.linalg.norm(X[i, :] - X[j, :])

    idx = np.argsort(Dist, axis=1)

    for i in range(0, n):
        for j in range(0, n):
            sigma_i = Dist[i, idx[i, 7]]
            sigma_j = Dist[j, idx[j, 7]]
            S_temp[i, j] = np.exp(-pow(Dist[i, j], 2) / (sigma_i * sigma_j))

    idx_new = idx[:, 0:6]

    S = np.zeros([n, n])
    for i in range(0, n):
        for j in range(1, 6):
            S[i, idx_new[i, j]] = S_temp[i, idx_new[i, j]]
            S[idx_new[i, j], i] = S_temp[i, idx_new[i, j]]

    while True:
        for i in range(0, n):
            for j in range(i, n):
                S[i, j] = S[i, j] / np.sum(S[i, :])
                S[j, i] = S[i, j]

        for i in range(0, n):
            for j in range(i, n):
                S[i, j] = S[i, j] / np.sum(S[:, j])
                S[j, i] = S[i, j]

        cond = np.linalg.norm(np.dot(S, np.ones((n, 1))) - np.ones((n, 1))) ** 2
        if cond < 1e-8:
            break

    S = (S + np.transpose(S)) / 2

    L = np.eye(n) - S

    return L