import numpy as np

def ridge(X, y, lmbda):
    '''
    RIDGE Ridge Regression.

      INPUT:  X: training sample features, P-by-N matrix.
              y: training sample labels, 1-by-N row vector.
              lmbda: regularization parameter.

      OUTPUT: w: learned parameters, (P+1)-by-1 column vector.

    NOTE: You can use pinv() if the matrix is singular.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    # YOUR CODE HERE
    # begin answer
    
    I=np.identity(P + 1)# Identity matrix
    X=np.vstack((np.ones((1,N)),X))
    w=np.dot(np.dot(np.linalg.pinv(np.dot(X,X.T)+lmbda*I),X),y.T)
    
    # end answer
    return w
