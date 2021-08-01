import numpy as np
from scipy.optimize import minimize

def svm(X, y):
    '''
    SVM Support vector machine.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
            num: number of support vectors

    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    num = 0

    # YOUR CODE HERE
    # Please implement SVM with scipy.optimize. You should be able to implement
    # it within 20 lines of code. The optimization should converge wtih any method
    # that support constrain.
    # begin answer
    def linear_constraint(w,X,y):#linear_constraint
        return y*np.dot(w.T,X)-1
    def minimize_object(w,X,y):#loss function
        return 0.5*np.sum(w**2)

    X=np.vstack((np.ones((1,N)),X))
    solver=minimize(minimize_object,w,args=(X,y),constraints=({'type': 'ineq', 'args': (X,y),'fun':lambda w,X,y:linear_constraint(w,X,y)}))
    w=solver.x    #w
    
    num=sum((y*np.dot(w.T,X)<1.1) * (y*np.dot(w.T,X)>0.1))#number of support vector
    # end answer
    return w, num

