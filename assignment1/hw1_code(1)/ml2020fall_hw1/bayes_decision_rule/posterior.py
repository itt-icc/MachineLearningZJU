import numpy as np
from likelihood import likelihood

def posterior(x):
    '''
    POSTERIOR Two Class Posterior Using Bayes Formula
    INPUT:  x, features of different class, C-By-N vector
            C is the number of classes, N is the number of different feature
    OUTPUT: p,  posterior of each class given by each feature, C-By-N matrix
    '''

    C, N = x.shape
    l = likelihood(x)
    total = np.sum(x)
    p = np.zeros((C, N))
    #TODO
    # begin answer
    for c in range(C):
        for n in range(N):
            p[c,n]=float((l[c,n]*(sum(x[c,:])/total))/(sum(x[:,n])/total))
    # end answer
    return p

