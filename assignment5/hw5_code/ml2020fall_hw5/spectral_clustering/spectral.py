import numpy as np


def spectral(W, k):
    '''
    SPECTRUAL spectral clustering

        Input:
            W: Adjacency matrix, N-by-N matrix
            k: number of clusters

        Output:
            idx: data point cluster labels, n-by-1 vector.
    '''
    # YOUR CODE HERE
    # begin answer
    D = np.diag(np.sum(W, axis = 0))
    L = D - W
    val, vec = np.linalg.eigh(L)#already sorted,eigh is more fast than eig
    #Use kmeans to clustering
    from kmeans import kmeans
    y=vec[:, :2]#only use the first  eignvector
    idx = kmeans(y, k)
    return idx
    # end answer
