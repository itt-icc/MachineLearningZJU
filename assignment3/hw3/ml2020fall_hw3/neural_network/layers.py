import numpy as np
from numpy.core.numeric import zeros_like
from fast_layers import *


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Args:
      x: (np.array) containing input data, of shape (N, d_1, ..., d_k)
      w: (np.array) weights, of shape (D, M)
      b: (np.array) biases, of shape (M,)

    Returns:
      out: output, of shape (N, M)
      cache: (x, w, b)
    """
    out = None
    #############################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You     #
    # will need to reshape the input into rows.                                 #
    #############################################################################
    # begin answer
    
    _x=x.reshape(x.shape[0],-1)#reshape x(N, d_1, ..., d_k)->_x(N,d_1 * ... * d_k)
    
    out=_x.dot(w)+b
    
    # end answer
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine layer.

    Args:
      dout: Upstream derivative, of shape (N, M)
      cache: Tuple of:
        x: Input data, of shape (N, d_1, ... d_k)
        w: Weights, of shape (D, M)

    Returns:
      dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
      dw: Gradient with respect to w, of shape (D, M)
      db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    #############################################################################
    # TODO: Implement the affine backward pass.                                 #
    #############################################################################
    # begin answer

    dx=(dout.dot(w.T)).reshape(x.shape)
    dw=x.reshape(x.shape[0],-1).T.dot(dout)
    db=np.ones((1,dout.shape[0])).dot(dout)
        
    # end answer
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Args:
      x: Inputs, of any shape

    Returns:
      out: Output, of the same shape as x
      cache: x
    """
    out = None
    #############################################################################
    # TODO: Implement the ReLU forward pass.                                    #
    #############################################################################
    # begin answer
    out=np.maximum(x, 0)
    # end answer
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Args:
      dout: Upstream derivatives, of any shape
      cache: Input x, of same shape as dout

    Returns:
      dx: Gradient with respect to x
    """
    dx, x = None, cache
    #############################################################################
    # TODO: Implement the ReLU backward pass.                                   #
    #############################################################################
    # begin answer
    
    dx=dout*np.where(x>0,1,0)
    
    # end answer
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.

    Args:
      x: Input data of shape (N, C, H, W)
      w: Filter weights of shape (F, C, HH, WW)
      b: Biases, of shape (F,)
      conv_param: A dictionary with the following keys:
        'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
        'pad': The number of pixels that will be used to zero-pad the input.

    Returns:
      out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
      cache: (x, w, b, conv_param)
    """
    out = None
    #############################################################################
    # TODO: Implement the convolutional forward pass.                           #
    # Hint: you can use the function np.pad for padding.                        #
    #############################################################################
    # begin answer
    pad=conv_param['pad']
    stride=conv_param['stride']
    new_H = int(1 + (x.shape[2] + 2 * pad - w.shape[2]) / stride)
    new_W = int(1 + (x.shape[3] + 2 * pad - w.shape[3]) / stride)
	#creat out
    out=np.zeros((x.shape[0],w.shape[0],new_H,new_W))
    
    #do padding
    x_pad=np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),'constant')
    
	#do conv
    for h_ in range(new_H):
        for w_ in range(new_W):
            local_area=x_pad[:,:,h_*stride:h_*stride+w.shape[2],w_*stride:w_*stride+w.shape[3]]
            for k in range(w.shape[0]):
                out[:,k,h_,w_]=np.sum(local_area*w[k,:,:,:],axis=(1,2,3))+b[k]#use matrix operetion for N
    # end answer
    cache = (x, w, b, conv_param)
    return out, cache




def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Args:
      dout: Upstream derivatives.
      cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns:
      dx: Gradient with respect to x
      dw: Gradient with respect to w
      db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    #############################################################################
    # TODO: Implement the convolutional backward pass.                          #
    #############################################################################
    # begin answer
    x,w,b,conv_param = cache
    N,C,H,W = x.shape
    F,_,HH,WW = w.shape
    stride,pad = conv_param['stride'],conv_param['pad']
    
    H_out = int(1+(H+2*pad-HH)//stride)
    W_out = int(1+(W+2*pad-WW)//stride)
    x_pad = np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),'constant')
    
    dw = np.zeros_like(w)
    dx = np.zeros_like(x)
    db = np.sum(dout,axis=(0,2,3))
    dx_pad = np.zeros_like(x_pad)
    
    for n in range(N):
        for i in range(H_out):
            for j in range(W_out):
                
                #I) for each kernal
                # for f in range(F):
                #     dw[f,:,:,:] += x_pad[n,:,i*stride:i*stride+HH,j*stride:j*stride+WW]*dout[n,f,i,j]
                #     dx_pad[n,:,i*stride:i*stride+HH,j*stride:j*stride+WW] += dout[n,f,i,j]*w[f,:,:,:]
                
                #II) some numpy trick for matrix operration to fast the operation
                dw += x_pad[n,:,i*stride:i*stride+HH,j*stride:j*stride+WW]*dout[n,:,i,j][:,None,None,None]#use matrix operetion for Channel,more fast
                dx_pad[n,:,i*stride:i*stride+HH,j*stride:j*stride+WW] += np.sum(dout[n,:,i,j][:,None,None,None]*w,axis=0)#use sum() to sum all the kernal   
    dx = dx_pad[:,:,pad:-pad,pad:-pad]
    
    # end answer
    return dx, dw, db




def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max pooling layer.

    Args:
      x: Input data, of shape (N, C, H, W)
      pool_param: dictionary with the following keys:
        'pool_height': The height of each pooling region
        'pool_width': The width of each pooling region
        'stride': The distance between adjacent pooling regions

    Returns:
      out: Output data
      cache: (x, pool_param)
    """
    out = None
    #############################################################################
    # TODO: Implement the max pooling forward pass                              #
    #############################################################################
    # begin answer
    N,C,H,W = x.shape
    pool_w,pool_h,stride=pool_param['pool_width'],pool_param['pool_height'],pool_param['stride']
    out_H=(H-pool_h)//stride+1
    out_W=(W-pool_w)//stride+1
    out=np.zeros((N,C,out_H,out_W))
    
    #I)for each sample and channel
    # for n in range(N):
    #     for c in range(C):
    #         for h in range(out_H):
    #             for w in range(out_W):
    #                 out[n,c,h,w]=np.max(x[n,c,h*stride:h*stride+pool_h,w*stride:w*stride+pool_w])
    
	#II)matrix operation
    for h in range(out_H):
        for w in range(out_W):
            out[:,:,h,w]=np.max(x[:,:,h*stride:h*stride+pool_h,w*stride:w*stride+pool_w],axis=(2,3))
                    
    
    # end answer
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max pooling layer.

    Args:
      dout: Upstream derivatives
      cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
      dx: Gradient with respect to x
    """
    dx = None
    #############################################################################
    # TODO: Implement the max pooling backward pass                             #
    #############################################################################
    # begin answer

    x, pool_param=cache
    N,C,H,W = x.shape
    pool_w,pool_h,stride=pool_param['pool_width'],pool_param['pool_height'],pool_param['stride']
    out_H=(H-pool_h)//stride+1
    out_W=(W-pool_w)//stride+1
    dx=np.zeros_like(x)
    
    #I)for each sample and channel
    for n in range(N):
        for c in range(C):
            for h in range(out_H):
                for w in range(out_W):
                    area=x[n,c,h*stride:h*stride+pool_h,w*stride:w*stride+pool_w]
                    index=np.unravel_index(area.argmax(), area.shape)
                    d=np.zeros_like(area)
                    d[index]=1
                    dx[n,c,h*stride:h*stride+pool_h,w*stride:w*stride+pool_w]+=d*dout[n,c,h,w]
    # end answer
    return dx


def svm_loss(x, y):
    """Computes the loss and gradient using for multiclass SVM classification.

    Args:
      x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
      y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns:
      loss: Scalar giving the loss
      dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    
    return loss, dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Args:
      x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
      y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns:
      loss: Scalar giving the loss
      dx: Gradient of the loss with respect to x
    """
    
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)# to probablity
    
    
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    
    # dL/dx=prob-1
    
    dx = probs.copy()
    dx[np.arange(N), y] -= 1 
    dx /= N
    
    return loss, dx
