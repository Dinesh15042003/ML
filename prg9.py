import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def kernel(point, xmat, k):
    m, n = xmat.shape
    weights = np.mat(np.eye(m))
    for j in range(m):
        diff = point - xmat[j]
        weights[j, j] = np.exp(diff.dot(diff.T) / (-2.0 * k**2))
    return weights
def local_weight(point, xmat, ymat, k):
    wei = kernel(point, xmat, k)
    XTWX = xmat.T.dot(wei).dot(xmat)
    XTWY = xmat.T.dot(wei).dot(ymat)
    W = np.linalg.pinv(XTWX).dot(XTWY)
    return W
def local_weight_regression(xmat, ymat, k):
    m, n = xmat.shape
    ypred = np.zeros(m)
    for i in range(m):
        W = local_weight(xmat[i], xmat, ymat, k)
        ypred[i] = xmat[i].dot(W)
    return ypred
data = pd.read_csv('tip.csv')
bill = np.array(data.total_bill)
tip = np.array(data.tip)
mbill = bill.reshape(-1, 1)
one = np.ones((mbill.shape[0], 1))
X = np.hstack((one, mbill)) 
k = 0.3
ypred = local_weight_regression(X, tip.reshape(-1, 1), k)
SortIndex = np.argsort(X[:, 1])
xsort = X[SortIndex, :]
ysort = ypred[SortIndex]
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(bill, tip, color='green')
ax.plot(xsort[:, 1], ysort, color='red', linewidth=5)
plt.xlabel('Total bill')
plt.ylabel('Tips')
plt.show()
