import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

# Helper functions to deal with image preprocessing
from cs231n.image_utils import load_image, preprocess_image, deprocess_image
from cs231n.classifiers.squeezenet import SqueezeNet

def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

# Older versions of scipy.misc.imresize yield different results
# from newer versions, so we check to make sure scipy is up to date.
def check_scipy():
    import scipy
    version = scipy.__version__.split('.')
    if int(version[0]) < 1:
        assert int(version[1]) >= 16, "You must install SciPy >= 0.16.0 to complete this notebook."

check_scipy()

# Load pretrained SqueezeNet model
SAVE_PATH = 'cs231n/datasets/squeezenet.ckpt'
if not os.path.exists(SAVE_PATH + ".index"):
    raise ValueError("You need to download SqueezeNet!")

model=SqueezeNet()
model.load_weights(SAVE_PATH)
model.trainable=False

# Load data for testing
content_img_test = preprocess_image(load_image('styles/tubingen.jpg', size=192))[None]
style_img_test = preprocess_image(load_image('styles/starry_night.jpg', size=192))[None]
answers = np.load('style-transfer-checks-tf.npz')

def content_loss(content_weight, content_current, content_original):
    """
    Compute the content loss for style transfer.
    
    Inputs:
    - content_weight: scalar constant we multiply the content_loss by.
    - content_current: features of the current image, Tensor with shape [1, height, width, channels]
    - content_target: features of the content image, Tensor with shape [1, height, width, channels]
    
    Returns:
    - scalar content loss
    """
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    n, h, w, c = tf.shape(content_current)
    cc = tf.reshape(tf.transpose(content_current[0], [2, 0, 1]), (c, h*w))
    co = tf.reshape(tf.transpose(content_original[0], [2, 0, 1]), (c, h*w))
    return content_weight * tf.reduce_sum(tf.math.square(tf.transpose(cc, (1, 0)) - tf.transpose(co, (1, 0))))

    n, h, w, c = content_current.shape
    cc = np.moveaxis(content_current, -1, 1).reshape(n, c, -1)
    co = np.moveaxis(content_original, -1, 1).reshape(n, c, -1)
    return content_weight * np.sum((np.moveaxis(cc, -1, 1) - np.moveaxis(co, -1, 1))**2) # strange that this gives a non-zero error

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# We provide this helper code which takes an image, a model (cnn), and returns a list of
# feature maps, one per layer.
def extract_features(x, cnn):
    """
    Use the CNN to extract features from the input image x.
    
    Inputs:
    - x: A Tensor of shape (N, H, W, C) holding a minibatch of images that
      will be fed to the CNN.
    - cnn: A Tensorflow model that we will use to extract features.
    
    Returns:
    - features: A list of feature for the input images x extracted using the cnn model.
      features[i] is a Tensor of shape (N, H_i, W_i, C_i); recall that features
      from different layers of the network may have different numbers of channels (C_i) and
      spatial dimensions (H_i, W_i).
    """
    features = []
    prev_feat = x
    for i, layer in enumerate(cnn.net.layers[:-2]):
        next_feat = layer(prev_feat)
        features.append(next_feat)
        prev_feat = next_feat
    return features

def content_loss_test(correct):
    content_layer = 2
    content_weight = 6e-2
    c_feats = extract_features(content_img_test, model)[content_layer]
    bad_img = tf.zeros(content_img_test.shape)
    feats = extract_features(bad_img, model)[content_layer]
    student_output = content_loss(content_weight, c_feats, feats)
    error = rel_error(correct, student_output)
    print('Maximum error is {:.3f}'.format(error))

content_loss_test(answers['cl_out'])

def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.
    
    Inputs:
    - features: Tensor of shape (1, H, W, C) giving features for
      a single image.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)
    
    Returns:
    - gram: Tensor of shape (C, C) giving the (optionally normalized)
      Gram matrices for the input image.
    """
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    n, h, w, c = tf.shape(features)
    f = tf.reshape(tf.transpose(features[0], [2, 0, 1]), (c, h*w))
    f = tf.transpose(f, [1, 0])
    G = tf.linalg.matmul(tf.transpose(f, [1, 0]), f)

    if normalize:
      G = G / tf.cast(h * w * c, tf.float32)
    
    return G

    co = tf.reshape(tf.transpose(content_original[0], [2, 0, 1]), (c, h*w))
    return content_weight * tf.reduce_sum(tf.math.square(tf.transpose(cc, (1, 0)) - tf.transpose(co, (1, 0))))


    n, h, w, c = features.shape
    f = np.moveaxis(features, -1, 1).reshape(n, c, -1)
    f = np.moveaxis(f, -1, 1)
    G = np.matmul(f[0].T, f[0])
    # G = np.empty((f.shape[2], f.shape[2]))
    # G[:] = np.nan
    # for i in range(f.shape[2]):
    #   for j in range(f.shape[2]):
    #     G[i, j] = np.sum(f[0][:, i]*f[0][:, j])
      
    if normalize:
      G = G / (h * w * c)
    
    return G

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

def gram_matrix_test(correct):
    gram = gram_matrix(extract_features(style_img_test, model)[4]) ### 4 instead of 5 - second MaxPooling layer
    error = rel_error(correct, gram)
    print('Maximum error is {:.3f}'.format(error))

gram_matrix_test(answers['gm_out'])

def style_loss(feats, style_layers, style_targets, style_weights):
    """
    Computes the style loss at a set of layers.
    
    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a Tensor giving the Gram matrix of the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].
      
    Returns:
    - style_loss: A Tensor containing the scalar style loss.
    """
    # Hint: you can do this with one for loop over the style layers, and should
    # not be short code (~5 lines). You will need to use your gram_matrix function.
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    L = 0
    for style_layer, style_target, style_weight in zip(style_layers, style_targets, style_weights):
      G = gram_matrix(feats[style_layer])
      L += style_weight * tf.reduce_sum(tf.math.square(G - style_target))
      # L += style_weight * np.sum((G - style_target)**2)

    return L

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

def style_loss_test(correct):
    style_layers = [0, 3, 5, 6]
    style_weights = [300000, 1000, 15, 3]
    
    c_feats = extract_features(content_img_test, model)
    feats = extract_features(style_img_test, model)
    style_targets = []
    for idx in style_layers:
        style_targets.append(gram_matrix(feats[idx]))
                             
    s_loss = style_loss(c_feats, style_layers, style_targets, style_weights)
    error = rel_error(correct, s_loss)
    print('Error is {:.3f}'.format(error))

style_loss_test(answers['sl_out'])

def tv_loss(img, tv_weight):
    """
    Compute total variation loss.
    
    Inputs:
    - img: Tensor of shape (1, H, W, 3) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    
    Returns:
    - loss: Tensor holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    # Your implementation should be vectorized and not require any loops!
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # shape_list = tf.shape(img)
    # H, W, C = shape_list[1], shape_list[2], shape_list[3]
    # sub_down = img[:,:H-1,:,:] - img[:,1:H,:,:]
    # sub_right = img[:,:,:W-1,:] - img[:,:,1:W,:]
    # return tv_weight * ( tf.reduce_sum(sub_down**2) + tf.reduce_sum(sub_right**2))

    tf.reduce_sum()

    return tv_weight * (tf.reduce_sum(tf.math.square(img[0][1:, ...] - img[0][:-1, ...])) + tf.reduce_sum(tf.math.square(img[0][:, 1:, ...] - img[0][:, :-1, ...])))
    return tv_weight * (np.sum(((img[0][1:, ...] - img[0][:-1, ...])**2)) + np.sum(((img[0][:, 1:, ...] - img[0][:, :-1, ...])**2)))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

def tv_loss_test(correct):
    tv_weight = 2e-2
    t_loss = tv_loss(content_img_test, tv_weight)
    error = rel_error(correct, t_loss)
    print('Error is {:.3f}'.format(error))

tv_loss_test(answers['tv_out'])

print('stall')