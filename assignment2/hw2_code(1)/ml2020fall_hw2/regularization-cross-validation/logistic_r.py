import numpy as np

def logistic_r(X, y, lmbda):
    '''
    LR Logistic Regression.

      INPUT:  X:   training sample features, P-by-N matrix.
              y:   training sample labels, 1-by-N row vector.
              lmbda: regularization parameter.

      OUTPUT: w    : learned parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    # YOUR CODE HERE
    # begin answer
    def sigmoid(inX):
    	return 1.0/(1+np.exp(-inX))
 
    X=np.vstack((np.ones((1,N)),X))
    y=np.where(y==-1,0,1)
    
    #SGD
    learning_rate=1e-4
    iter=0
    while True:
        iter+=1
        _w=w.copy()
        
        for i in range(N):
            error=y[0,i]-sigmoid(np.dot(w[:,0].T,X[:,i]))    #compute the loss

            w[:,0]+=learning_rate*(error*X[:,i]-lmbda*w[:,0])#update w use SGD
            
        # Use infinite norm constraints
        if max(abs(_w-w))<=1e-4:
            break
        
        #early drop to save time
        if iter==200:
            # print('warning : out of iter')
            break
        
    # end answer
    return w
