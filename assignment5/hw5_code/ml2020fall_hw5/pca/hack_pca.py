import numpy as np
import matplotlib.pyplot as plt
from pca import *

def hack_pca(filename):
    '''
    Input: filename -- input image file name/path
    Output: img -- image without rotation
    '''
    img_r = (plt.imread(filename)).astype(np.float64)
    plt.imshow(img_r.astype(np.uint8))
    plt.show()
    # YOUR CODE HERE
    # begin answer
    #get coordate point
    data_point=np.array([np.array([i,j])  for i in range(img_r.shape[0]) for j in range(img_r.shape[1]) if np.sum(img_r[i,j,:])!=0])
    eigvector,eigvalue=PCA(data_point)
    new_data_point=(data_point.dot(eigvector)).astype(int)
    
    new_data_point-=np.min(new_data_point,axis=0)
    size=np.max(new_data_point,axis=0)+1
    img=np.zeros((size[0],size[1],img_r.shape[2]))
    
    for i in range(data_point.shape[0]):
        img[new_data_point[i,:][0],new_data_point[i,:][1],:]=img_r[data_point[i,:][0],data_point[i,:][1],:]
    return np.sum(img,axis=2)#to gray image
    # end answer