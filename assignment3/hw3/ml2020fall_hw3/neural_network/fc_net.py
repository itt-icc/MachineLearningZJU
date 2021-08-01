import numpy as np
from layers import *
from layer_utils import *

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=1 * 28 * 28, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Args:
          input_dim: An integer giving the size of the input
          hidden_dim: An integer giving the size of the hidden layer
          num_classes: An integer giving the number of classes to classify
          weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
          reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        # begin answer
        self.params['W1']=np.random.normal(loc=0,scale=weight_scale,size=(input_dim,hidden_dim))
        self.params['b1']=np.zeros(hidden_dim)
        self.params['W2']=np.random.normal(loc=0,scale=weight_scale,size=(hidden_dim,num_classes))
        self.params['b2']=np.zeros(num_classes)
        # end answer

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Args:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # begin answer
        
        #forward pass
        outf1,f1_cache=affine_forward(X, self.params['W1'], self.params['b1'])
        out_relu1,relu1_cache=relu_forward(outf1)
        outf2,fc2_cache=affine_forward(out_relu1,self.params['W2'],self.params['b2'])
        
        
        if y is None:
            scores=outf2
            return scores

        #softmax loss
        soft_loss,dx=softmax_loss(outf2,y)
        #L2 reg loss
        reg_loss=0.5 * self.reg * np.sum(self.params['W1']**2) + 0.5 * self.reg * np.sum(self.params['W2']**2)
        loss=soft_loss+reg_loss
        
        #backward pass
        dx, dw2, db2 = affine_backward(dx, fc2_cache)
        dx=relu_backward(dx,relu1_cache)
        dx, dw1, db1 = affine_backward(dx, f1_cache)
        grads={"W1":dw1+self.params['W1']*self.reg,'W2':dw2+self.params['W2']*self.reg,'b1':db1,'b2':db2}
        
        # end answer
        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function.
    For a network with L layers, the architecture will be:

    {affine - relu} x (L - 1) - affine - softmax

    where the {...} block is repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=1 * 28 * 28, num_classes=10,
                 reg=0.0, weight_scale=1e-2,
                 dtype=np.float32):
        """
        Initialize a new FullyConnectedNet.

        Args:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        """
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        ############################################################################
        # begin answer
        for i in range(self.num_layers):
            if i==0:#first hidden layer
                self.params['W'+str(i+1)]=np.random.normal(loc=0,scale=weight_scale,size=(input_dim,hidden_dims[i]))
                self.params['b'+str(i+1)]=np.zeros(hidden_dims[i])
            elif i==self.num_layers-1:#last hidden layers
                self.params['W'+str(i+1)]=np.random.normal(loc=0,scale=weight_scale,size=(hidden_dims[i-1],num_classes))
                self.params['b'+str(i+1)]=np.zeros(num_classes)
            else:
                self.params['W'+str(i+1)]=np.random.normal(loc=0,scale=weight_scale,size=(hidden_dims[i-1],hidden_dims[i]))
                self.params['b'+str(i+1)] = np.zeros(hidden_dims[i])

        # end answer

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
            # print('%s: ' % k, v.shape)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        ############################################################################
        # begin answer
        dict_cache={}#to store the cache
        scores=X.copy()
        for i in range(self.num_layers-1):
            scores,dict_cache[str(i+1)]=affine_relu_forward(scores, self.params['W'+str(i+1)], self.params['b'+str(i+1)])
        scores,dict_cache[str(self.num_layers)]=affine_forward(scores, self.params['W'+str(self.num_layers)], self.params['b'+str(self.num_layers)])
        # end answer
        # If test mode return early
        if y is None:
            return scores
        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # begin answer
        #softmax loss
        soft_loss,dx=softmax_loss(scores,y)
        #L2 reg loss
        reg_loss=0.5 * self.reg * sum([np.sum(self.params.get(i,0)**2) for i in self.params.keys() if i[0]=='W'])
        #total loss
        loss=soft_loss+reg_loss
        #output layer
        dx,grads['W'+str(self.num_layers)],grads['b'+str(self.num_layers)]=affine_backward(dx,dict_cache[str(self.num_layers)])
        grads['W'+str(self.num_layers)]+=self.params['W'+str(self.num_layers)]*self.reg
        #hidden layer
        for i in range(self.num_layers-1,0,-1):
            dx,grads['W'+str(i)],grads['b'+str(i)]=affine_relu_backward(dx,dict_cache[str(i)])
            grads['W'+str(i)]+=self.params['W'+str(i)]*self.reg
        # end answer
        return loss, grads