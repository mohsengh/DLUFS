import numpy as np
import math
import scipy


def dlufs(X, **kwargs):
    alpha = kwargs['alpha']
    lambd = kwargs['lambd']
    L = kwargs['L']
    r = kwargs['r']

    X = X.T
    p, n = X.shape

    Z = X

    maxIter = 100
    obj = np.zeros(maxIter)

    for iter_step in range(maxIter):

        temp = np.sqrt((Z * Z).sum(1))
        temp[temp < 1e-16] = 1e-16
        temp = 0.5 / temp
        D = np.diag(temp)

        St = np.dot(Z, Z.T) + 1e-6 * np.identity(p)
        St_inv = scipy.linalg.inv(St)
        ZX = np.dot(Z, X.T)
        Sb = np.dot(ZX, ZX.T)
        T1 = np.dot(St_inv, Sb)
        W, V = scipy.linalg.eig(T1)
        eigenValues = np.real(W)
        eigenVectors = np.real(V)

        idx = (-eigenValues).argsort()
        eigenVectors = eigenVectors[:, idx[0:r]]
        B = eigenVectors.T

        BSB = np.dot(B, np.dot(St, B.T)) + 1e-6 * np.identity(r)
        XZB = np.dot(X, np.dot(Z.T, B.T))
        A = np.dot(XZB, np.linalg.inv(BSB))

        AB = np.dot(A, B)

        a = np.dot(AB.T, AB) + lambd * D
        b = alpha * L
        q = np.dot(AB.T, X)
        Z = scipy.linalg.solve_sylvester(a, b, q)

        ZLZ = np.dot(Z, np.dot(L, Z.T))
        Z_2_1 = (np.sqrt((Z*Z).sum(1))).sum()
        ABZ = np.dot(AB, Z)
        obj[iter_step] = np.linalg.norm(X - ABZ, 'fro')**2 + alpha*np.trace(ZLZ) + lambd*Z_2_1

        if iter_step >= 1 and math.fabs(obj[iter_step] - obj[iter_step - 1]) / math.fabs(obj[iter_step]) < 1e-3:
            break

    print iter_step
    return Z
