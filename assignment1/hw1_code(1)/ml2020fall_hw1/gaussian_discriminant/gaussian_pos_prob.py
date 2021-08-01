import numpy as np

def gaussian_pos_prob(X, Mu, Sigma, Phi):
    '''
    GAUSSIAN_POS_PROB Posterior probability of GDA.
    Compute the posterior probability of given N data points X
    using Gaussian Discriminant Analysis where the K gaussian distributions
    are specified by Mu, Sigma and Phi.
    Inputs:
        'X'     - M-by-N numpy array, N data points of dimension M.
        'Mu'    - M-by-K numpy array, mean of K Gaussian distributions.
        'Sigma' - M-by-M-by-K  numpy array (yes, a 3D matrix), variance matrix of
                  K Gaussian distributions.
        'Phi'   - 1-by-K  numpy array, prior of K Gaussian distributions.
    Outputs:
        'p'     - N-by-K  numpy array, posterior probability of N data points
                with in K Gaussian distribsubplots_adjustutions.
    ''' 
    N = X.shape[1]
    K = Phi.shape[0]
    p = np.zeros((N, K))
    #Your code HERE

    # begin answer
    
    
    import time
    start=time.time()
    #compute likelihood
    inv_Sigama=np.zeros(Sigma.shape)
    for k in range(K):inv_Sigama[:,:,k]=np.linalg.inv(Sigma[:,:,k])
    
    Expo=np.zeros((N, K))
    for n in range(N):
        for k in range(K):
            Expo[n,k]=-0.5*np.dot(np.dot((X[:,n]-Mu[:,k]).T,inv_Sigama[:,:,k]),(X[:,n]-Mu[:,k]))
            
    neg_Sigama=np.zeros((N,K))
    for k in range(K):neg_Sigama[:,k]=np.linalg.det(Sigma[:,:,k])
    
    l= np.zeros((N, K))
    for k in range(K):l[:,k]=np.exp(Expo[:,k])/(2*3.1415926*np.power(neg_Sigama[:,k],0.5))
    
    #compute Evidence
    Evidence=np.ones(N)
    for n in range(N):Evidence[n]=sum(l[n,:]*Phi)
    
    #compute posterior
    for k in range(K):p[:,k]=l[:,k]*Phi[k]/Evidence
    end=time.time()
    print("Posterior time cost = {}S".format(end-start))
    # end answer
    
    return p
    