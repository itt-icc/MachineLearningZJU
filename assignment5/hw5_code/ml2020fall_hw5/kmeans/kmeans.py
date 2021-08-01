import numpy as np


def kmeans(x, k):
    '''
    KMEANS K-Means clustering algorithm

        Input:  x - data point features, n-by-p maxtirx.
                k - the number of clusters

        OUTPUT: idx  - cluster label
                ctrs - cluster centers, K-by-p matrix.
                iter_ctrs - cluster centers of each iteration, (iter, k, p)
                        3D matrix.
    '''
    # YOUR CODE HERE

    # begin answer
    n = x.shape[0]
    row = np.arange(n)
    np.random.shuffle(row)
    ctrs = x[row[:k]]
    iter_ctrs = [ctrs]
    idx = np.ones(n)
    x_k = np.repeat(x[:, :, np.newaxis], k, axis=2)  # vector compute
    
    maxiter=0
    while True and maxiter!=1000:
        maxiter+=1
        
        dis = np.sum((x_k-ctrs.T)**2, axis=1) #broadcast
        
        new_idx = np.argmin(dis, axis=1)
        if (new_idx == idx).all():break
        ctrs=np.ones(ctrs.shape)
        for i in range(k):ctrs[i] = np.mean(x[new_idx == i], axis=0)
        iter_ctrs.append(ctrs)
        idx = new_idx 

    iter_ctrs = np.array(iter_ctrs)
    # end answer

    return idx, ctrs, iter_ctrs
