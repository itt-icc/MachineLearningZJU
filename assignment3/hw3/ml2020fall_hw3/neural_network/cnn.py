import numpy as np

from layers import *
from fast_layers import *
from layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(1, 28, 28), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.bn_params = {}
        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        # begin answer
        self.params['W1']=np.random.normal(loc=0,scale=weight_scale,size=(num_filters,input_dim[0],filter_size,filter_size))
        self.params['b1']=np.zeros(num_filters)
        
        self.params['W2']=np.random.normal(loc=0,scale=weight_scale,size=(num_filters*input_dim[1]*input_dim[1]//4,hidden_dim))
        self.params['b2']=np.zeros(hidden_dim)
        
        self.params['W3']=np.random.normal(loc=0,scale=weight_scale,size=(hidden_dim,num_classes))
        self.params['b3']=np.zeros(num_classes)
        # end answer

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        N = X.shape[0]

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None

        X = X.astype(self.dtype)

        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        # begin answer
        
        out1,cache1=conv_relu_pool_forward(X,W1,b1,conv_param,pool_param)
        out2,cache2=affine_relu_forward(out1,W2,b2)
        scores,cache3=affine_forward(out2,W3,b3)
        
        # end answer

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        # begin answer

        soft_loss,dx=softmax_loss(scores,y)
        reg_loss=0.5 * self.reg * sum([np.sum(self.params.get(i,0)**2) for i in self.params.keys() if i[0]=='W'])
        loss=soft_loss+reg_loss
        
        dx,grads['W3'],grads['b3']=affine_backward(dx,cache3)
        dx,grads['W2'],grads['b2']=affine_relu_backward(dx,cache2)
        dx,grads["W1"],grads['b1']=conv_relu_pool_backward(dx,cache1)
        
        for i in range(1,4):
            grads['W'+str(i)]+=self.params['W'+str(i)]*self.reg
            
        # end answer
        return loss, grads



class ConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(1, 28, 28), num_filters1=32, filter_size1=3,num_filters2=32, filter_size2=3,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):

        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.bn_params = {}
        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        # begin answer
        self.params['W1']=np.random.normal(loc=0,scale=weight_scale,size=(num_filters1,input_dim[0],filter_size1,filter_size1))
        self.params['b1']=np.zeros(num_filters1)
        
        self.params['W11']=np.random.normal(loc=0,scale=weight_scale,size=(num_filters2,num_filters1,filter_size2,filter_size2))
        self.params['b11']=np.zeros(num_filters2)
        
        self.params['W2']=np.random.normal(loc=0,scale=weight_scale,size=(num_filters2*input_dim[1]*input_dim[1]//16,hidden_dim))
        self.params['b2']=np.zeros(hidden_dim)
        
        self.params['W3']=np.random.normal(loc=0,scale=weight_scale,size=(hidden_dim,num_classes))
        self.params['b3']=np.zeros(num_classes)
        # end answer

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
    # def cal_dim(x):
        
        

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        N = X.shape[0]

        W1, b1 = self.params['W1'], self.params['b1']
        W11, b11 = self.params['W11'], self.params['b11']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None

        X = X.astype(self.dtype)

        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        # begin answer
        
        out1,cache1=conv_relu_pool_forward(X,W1,b1,conv_param,pool_param)
        out11,cache11=conv_relu_pool_forward(out1,W11,b11,conv_param,pool_param)
        out2,cache2=affine_relu_forward(out11,W2,b2)
        scores,cache3=affine_forward(out2,W3,b3)
        
        # end answer

        if y is None:
            return scores

        loss, grads = 0, {}
        # begin answer

        soft_loss,dx=softmax_loss(scores,y)
        reg_loss=0.5 * self.reg * sum([np.sum(self.params.get(i,0)**2) for i in self.params.keys() if i[0]=='W'])
        loss=soft_loss+reg_loss
        
        dx,grads['W3'],grads['b3']=affine_backward(dx,cache3)
        dx,grads['W2'],grads['b2']=affine_relu_backward(dx,cache2)
        dx,grads["W11"],grads['b11']=conv_relu_pool_backward(dx,cache11)
        dx,grads["W1"],grads['b1']=conv_relu_pool_backward(dx,cache1)
        
        for i in range(1,4):
            grads['W'+str(i)]+=self.params['W'+str(i)]*self.reg
        grads['W11']+=self.params['W11']*self.reg
        # end answer
        return loss, grads