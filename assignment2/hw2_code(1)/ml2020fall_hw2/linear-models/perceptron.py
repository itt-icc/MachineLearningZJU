from gc import get_threshold
import numpy as np

def perceptron(X, y):
    '''
    PERCEPTRON Perceptron Learning Algorithm.

       INPUT:  X: training sample features, P-by-N matrix.
               y: training sample labels, 1-by-N row vector.

       OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
               iter: number of iterations
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    iters = 0
    # YOUR CODE HERE
    # begin answer

    X=np.vstack((np.ones((1,N)),X))
    while True:
        iters+=1
        _w=w.copy()
        for i in range(N):
            pre_y=np.where(np.dot(w[:,0].T,X[:,i])>0,1,-1)
            if pre_y*y[0,i]<0:
                w[:,0]+=y[0,i]*X[:,i]
        # use Infinity normal value
        if max(np.abs(_w-w))<=1e-4:
            break
        
        if iters==200:
            break

    # end answer
    return w, iters