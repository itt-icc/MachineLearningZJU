import numpy as np

def knn_graph(X, k, threshold):
    '''
    KNN_GRAPH Construct W using KNN graph

        Input:
            X - data point features, n-by-p maxtirx.
            k - number of nn.
            threshold - distance threshold.

        Output:
            W - adjacency matrix, n-by-n matrix.
    '''

    # YOUR CODE HERE
    # begin answer
    n=X.shape[0]
    W=np.zeros((n,n))
    for i in range(n):
        item = np.ones((n,1)).dot(np.expand_dims(X[i, :], axis=0))#to fast the distance compute
        distance= np.sum(np.square(item-X), axis=1)
        index = np.argsort(distance)[1:k+1]
        #Use threshold
        new_idx=[i for i in index if distance[i]<=threshold]
        #build ajacent matrix
        W[i,new_idx]=1
        W[new_idx,i]=1
    return W
    # end answer

