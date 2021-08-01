import numpy as np
from scipy.optimize import minimize


def svm(X, y, C=0.001):

    P, N = X.shape
    X = np.vstack((np.ones((1, N)), X))
    W_Beta = np.ones(P+N+1)
    c = C
    relaxation_type = 'eq' if C == 0 else 'ineq'
    num = 0

    def linear_constraint(W_Beta, X, y):#linear constraint
        w = W_Beta[:P+1]
        beta = W_Beta[P+1:]
        return y*w.T.dot(X)+beta-1

    def minimize_object(W_Beta, X, y, c):#loss function
        w = W_Beta[:P+1]
        beta = W_Beta[P+1:]
        object_value = 0.5*np.sum(w**2)+c*np.sum(beta)
        return object_value

    def relaxation_constraint(W_Beta, y):#slack
        beta = W_Beta[P+1:]
        return beta

    solver = minimize(minimize_object, W_Beta, args=(X, y, c), 
        constraints=(
        {'type': 'ineq', 'args': (X, y), 'fun': lambda w,X, y: linear_constraint(w, X, y)},
        {'type': relaxation_type, 'args': (y,), 'fun': lambda W_Beta, y: relaxation_constraint(W_Beta, y)}
    ))
    w = solver.x[:P+1] #w
    beta = solver.x[P+1:]#beta
    num = sum((y*np.dot(w.T, X) <= 1 - beta))#number of support vector
    return w, num
