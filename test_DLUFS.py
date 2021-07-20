import numpy as np
import scipy.io
from DLUFS import dlufs
from construct_L import construct_L

def main():
    # load data
    data_name = 'Yale'
    print data_name

    mat = scipy.io.loadmat(data_name + '.mat')
    X = mat['X']
    X = X.astype(float)
    y = mat['Y']
    Parm = [1e-4, 1e-2, 1, 1e+2, 1e+4]

    p = len(X[0])
    n = len(X[:, 0])

    step = 50
    num_fea = 300

    X = (X - X.mean(axis=0)) / X.std(axis=0)

    L = construct_L(X)

    idx = np.zeros((p, 25, 6), dtype=np.int)

    for r in range(step, num_fea + 1, step):
        print r
        count = 0
        for Parm1 in Parm:
            for Parm2 in Parm:
                print r
                print count
                Weight = dlufs(X, L=L, r=r, alpha=Parm1, lambd=Parm2)
                idx[0:p, count, (r/step)-1] = feature_ranking(Weight)
                count += 1


def feature_ranking(W):
    """
    This function ranks features according to the feature weights matrix W

    Input:
    -----
    W: {numpy array}, shape (n_features, n_classes)
        feature weights matrix

    Output:
    ------
    idx: {numpy array}, shape {n_features,}
        feature index ranked in descending order by feature importance
    """
    T = (W*W).sum(1)
    idx = np.argsort(T, 0)
    return idx[::-1]


if __name__ == '__main__':
    main()
