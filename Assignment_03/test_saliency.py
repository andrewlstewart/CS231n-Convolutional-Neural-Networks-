import sys
sys.path.append('../')
import time, os, json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from cs231n.classifiers.squeezenet import SqueezeNet
from cs231n.data_utils import load_tiny_imagenet
from cs231n.image_utils import preprocess_image, deprocess_image
from cs231n.image_utils import SQUEEZENET_MEAN, SQUEEZENET_STD

SAVE_PATH = 'cs231n/datasets/squeezenet.ckpt'

if not os.path.exists(SAVE_PATH + ".index"):
    raise ValueError("You need to download SqueezeNet!")

model = SqueezeNet()
status = model.load_weights(SAVE_PATH)

model.trainable = False

from cs231n.data_utils import load_imagenet_val
X_raw, y, class_names = load_imagenet_val(num=5)

# plt.figure(figsize=(12, 6))
# for i in range(5):
#     plt.subplot(1, 5, i + 1)
#     plt.imshow(X_raw[i])
#     plt.title(class_names[y[i]])
#     plt.axis('off')
# plt.gcf().tight_layout()
# plt.show()

X = np.array([preprocess_image(img) for img in X_raw])

def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images, numpy array of shape (N, H, W, 3)
    - y: Labels for X, numpy of shape (N,)
    - model: A SqueezeNet model that will be used to compute the saliency map.

    Returns:
    - saliency: A numpy array of shape (N, H, W) giving the saliency maps for the
    input images.
    """
    saliency = None
    # Compute the score of the correct class for each example.
    # This gives a Tensor with shape [N], the number of examples.
    #
    # Note: this is equivalent to scores[np.arange(N), y] we used in NumPy
    # for computing vectorized losses.
    
    ###############################################################################
    # TODO: Produce the saliency maps over a batch of images.                     #
    #                                                                             #
    # 1) Define a gradient tape object and watch input Image variable             #
    # 2) Compute the “loss” for the batch of given input images.                  #
    #    - get scores output by the model for the given batch of input images     #
    #    - use tf.gather_nd or tf.gather to get correct scores                    #
    # 3) Use the gradient() method of the gradient tape object to compute the     #
    #    gradient of the loss with respect to the image                           #
    # 4) Finally, process the returned gradient to compute the saliency map.      #
    ###############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N = X.shape[0]
    X = tf.convert_to_tensor(X)
    with tf.GradientTape() as tape:
        # TODO: Understand why you need the 'correct_scores' and if you do need them, understand how the loss then gets calculated...
        tape.watch(X)
        scores = model(X) # model.predict returns a numpy array which can't contain the gradient information https://github.com/tensorflow/tensorflow/issues/20630
        soft_max = tf.nn.softmax(scores)
        correct_scores = tf.gather_nd(soft_max, tf.stack((tf.range(N), y), axis=1))
        # loss = tf.keras.losses.sparse_categorical_crossentropy(y, soft_max)
        loss = tf.nn.softmax_cross_entropy(y, correct_scores)

        # soft_max = tf.nn.softmax(scores)
        
        # target_y_col = tf.repeat(tf.convert_to_tensor(target_y), soft_max.shape[0])
        # # https://stackoverflow.com/questions/48406763/how-to-index-and-assign-to-a-tensor-in-tensorflow
        # # max_inds = tf.argmax(soft_max, axis=1)
        # shape = soft_max.get_shape()
        # inds = tf.range(0, shape[1], dtype=target_y_col.dtype)[None, :]
        # bmask = tf.equal(inds, target_y_col[:, None])
        # imask = tf.where(bmask, tf.ones_like(soft_max), tf.zeros_like(soft_max))
        # newmat = soft_max * imask

        # loss = tf.keras.losses.sparse_categorical_crossentropy(target_y_col, newmat)
        # grad = tape.gradient(loss, X_fooling)

        # tf.gather_nd(soft_max, tf.stack((tf.range(N), target_y), axis=1))
        # correct_scores = tf.zeros_like(scores)
        # correct_scores = tf.where(bmask, )

    grad = tape.gradient(loss, X)
    saliency = tf.math.reduce_max(tf.math.abs(grad), axis=-1)
        
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return saliency

def show_saliency_maps(X, y, mask):
    mask = np.asarray(mask)
    Xm = X[mask]
    ym = y[mask]

    saliency = compute_saliency_maps(Xm, ym, model)

    for i in range(mask.size):
        plt.subplot(2, mask.size, i + 1)
        plt.imshow(deprocess_image(Xm[i]))
        plt.axis('off')
        plt.title(class_names[ym[i]])
        plt.subplot(2, mask.size, mask.size + i + 1)
        plt.title(mask[i])
        plt.imshow(saliency[i], cmap=plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(10, 4)
    plt.show()

# mask = np.arange(5)
# show_saliency_maps(X, y, mask)

def make_fooling_image(X, target_y, model):
    """
    Generate a fooling image that is close to X, but that the model classifies
    as target_y.

    Inputs:
    - X: Input image, a numpy array of shape (1, 224, 224, 3)
    - target_y: An integer in the range [0, 1000)
    - model: Pretrained SqueezeNet model

    Returns:
    - X_fooling: An image that is close to X, but that is classifed as target_y
    by the model.
    """
    
    # Make a copy of the input that we will modify
    X_fooling = X.copy()
    
    # Step size for the update
    learning_rate = 1
    
    ##############################################################################
    # TODO: Generate a fooling image X_fooling that the model will classify as   #
    # the class target_y. Use gradient *ascent* on the target class score, using #
    # the model.scores Tensor to get the class scores for the model.image.   #
    # When computing an update step, first normalize the gradient:               #
    #   dX = learning_rate * g / ||g||_2                                         #
    #                                                                            #
    # You should write a training loop, where in each iteration, you make an     #
    # update to the input image X_fooling (don't modify X). The loop should      #
    # stop when the predicted class for the input is the same as target_y.       #
    #                                                                            #
    # HINT: Use tf.GradientTape() to keep track of your gradients and            #
    # use tape.gradient to get the actual gradient with respect to X_fooling.    #
    #                                                                            #
    # HINT 2: For most examples, you should be able to generate a fooling image  #
    # in fewer than 100 iterations of gradient ascent. You can print your        #
    # progress over iterations to check your algorithm.                          #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    assert X.shape[0] == 1, 'This function will only work for a single input'
    X_fooling = tf.convert_to_tensor(X_fooling)
    for _ in range(100):
        with tf.GradientTape() as tape:
            tape.watch(X_fooling)
            score = model(X_fooling)
            if tf.argmax(score) == target_y:
                break
            correct_score = score[0, target_y]

        grad = tape.gradient(correct_score, X_fooling)
        dX = learning_rate * tf.math.l2_normalize(grad)
        X_fooling += dX # Ascent

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return X_fooling

idx = 0
Xi = X[idx][None]
target_y = 6
X_fooling = make_fooling_image(Xi, target_y, model)

# Make sure that X_fooling is classified as y_target
scores = model(X_fooling)
assert tf.math.argmax(scores[0]).numpy() == target_y, 'The network is not fooled!'

# # Show original image, fooling image, and difference
# orig_img = deprocess_image(Xi[0])
# fool_img = deprocess_image(X_fooling[0])
# plt.figure(figsize=(12, 6))

# # Rescale 
# plt.subplot(1, 4, 1)
# plt.imshow(orig_img)
# plt.axis('off')
# plt.title(class_names[y[idx]])
# plt.subplot(1, 4, 2)
# plt.imshow(fool_img)
# plt.title(class_names[target_y])
# plt.axis('off')
# plt.subplot(1, 4, 3)
# plt.title('Difference')
# plt.imshow(deprocess_image((Xi-X_fooling)[0]))
# plt.axis('off')
# plt.subplot(1, 4, 4)
# plt.title('Magnified difference (10x)')
# plt.imshow(deprocess_image(10 * (Xi-X_fooling)[0]))
# plt.axis('off')
# plt.gcf().tight_layout()
# plt.show()

from scipy.ndimage.filters import gaussian_filter1d
def blur_image(X, sigma=1):
    X = gaussian_filter1d(X, sigma, axis=1)
    X = gaussian_filter1d(X, sigma, axis=2)
    return X

def jitter(X, ox, oy):
    """
    Helper function to randomly jitter an image.
    
    Inputs
    - X: Tensor of shape (N, H, W, C)
    - ox, oy: Integers giving number of pixels to jitter along W and H axes
    
    Returns: A new Tensor of shape (N, H, W, C)
    """
    if ox != 0:
        left = X[:, :, :-ox]
        right = X[:, :, -ox:]
        X = tf.concat([right, left], axis=2)
    if oy != 0:
        top = X[:, :-oy]
        bottom = X[:, -oy:]
        X = tf.concat([bottom, top], axis=1)
    return X

def create_class_visualization(target_y, model, **kwargs):
    """
    Generate an image to maximize the score of target_y under a pretrained model.
    
    Inputs:
    - target_y: Integer in the range [0, 1000) giving the index of the class
    - model: A pretrained CNN that will be used to generate the image
    
    Keyword arguments:
    - l2_reg: Strength of L2 regularization on the image
    - learning_rate: How big of a step to take
    - num_iterations: How many iterations to use
    - blur_every: How often to blur the image as an implicit regularizer
    - max_jitter: How much to jitter the image as an implicit regularizer
    - show_every: How often to show the intermediate result
    """
    l2_reg = kwargs.pop('l2_reg', 1e-3)
    learning_rate = kwargs.pop('learning_rate', 25)
    num_iterations = kwargs.pop('num_iterations', 100)
    blur_every = kwargs.pop('blur_every', 10)
    max_jitter = kwargs.pop('max_jitter', 16)
    show_every = kwargs.pop('show_every', 25)
    
    # We use a single image of random noise as a starting point
    X = 255 * np.random.rand(224, 224, 3)
    X = preprocess_image(X)[None]

    loss = None # scalar loss
    grad = None # gradient of loss with respect to model.image, same size as model.image
    
    X = tf.Variable(X)
    for t in range(num_iterations):
        # Randomly jitter the image a bit; this gives slightly nicer results
        ox, oy = np.random.randint(0, max_jitter, 2)
        X = jitter(X, ox, oy)
        
        ########################################################################
        # TODO: Compute the value of the gradient of the score for             #
        # class target_y with respect to the pixels of the image, and make a   #
        # gradient step on the image using the learning rate. You should use   #
        # the tf.GradientTape() and tape.gradient to compute gradients.        #
        #                                                                      #
        # Be very careful about the signs of elements in your code.            #
        ########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        with tf.GradientTape() as tape:
            tape.watch(X)
            score = model(X)
            correct_score = score[0, target_y]
            img = correct_score - l2_reg * tf.nn.l2_normalize(X)
        
        grad = tape.gradient(img, X)
        dX = learning_rate * tf.math.l2_normalize(grad)
        X += dX

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        
        # Undo the jitter
        X = jitter(X, -ox, -oy)
        # As a regularizer, clip and periodically blur
        
        X = tf.clip_by_value(X, -SQUEEZENET_MEAN/SQUEEZENET_STD, (1.0 - SQUEEZENET_MEAN)/SQUEEZENET_STD)
        if t % blur_every == 0:
            X = blur_image(X, sigma=0.5)

        # Periodically show the image
        if t == 0 or (t + 1) % show_every == 0 or t == num_iterations - 1:
            plt.imshow(deprocess_image(X[0]))
            class_name = class_names[target_y]
            plt.title('%s\nIteration %d / %d' % (class_name, t + 1, num_iterations))
            plt.gcf().set_size_inches(4, 4)
            plt.axis('off')
            plt.show()
    return X

# target_y = 76 # Tarantula
# out = create_class_visualization(target_y, model)

target_y = np.random.randint(1000)
# target_y = 78 # Tick
target_y = 187 # Yorkshire Terrier
# target_y = 683 # Oboe
# target_y = 366 # Gorilla
# target_y = 604 # Hourglass
print(class_names[target_y])
X = create_class_visualization(target_y, model)

print('stall')