import numpy as np
import scipy.stats


def knn(x, x_train, y_train, k):
    '''
    KNN k-Nearest Neighbors Algorithm.

        INPUT:  x:         testing sample features, (N_test, P) matrix.
                x_train:   training sample features, (N, P) matrix.
                y_train:   training sample labels, (N, ) column vector.
                k:         the k in k-Nearest Neighbors

        OUTPUT: y    : predicted labels, (N_test, ) column vector.
    '''

    # Warning: uint8 matrix multiply uint8 matrix may cause overflow, take care
    # Hint: You may find numpy.argsort & scipy.stats.mode helpful
    # Hint: distance = (x - x_train)^2. How to vectorize this?

    # YOUR CODE HERE
    # begin answer
    y = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        item = np.ones((x_train.shape[0],1)).dot(np.expand_dims(x[i, :], axis=0))
        distance = np.sum(np.square(item-x_train), axis=1)
        index = np.argsort(distance)[:k]
        result=scipy.stats.mode(y_train[index])
        y[i]=result[0][0]
    # end answer

    return y
