"""
Utils for Art Generation with Neural Style Transfer (NST):
"""

import numpy as np
import tensorflow as tf
from PIL import Image

import tensorflow as tf


def clip_values_0_1(image):
    """
    Clips all tensor pixel values to be within the 0 to 1 range. This helps stabilize the training process,
    prevent artifacts, and ensure the output image is displayable and visually coherent.

    Without clipping, some pixels might take extremely high or low values, leading to unintended visual artifacts
    and vanishing/exploding gradients.

    Arguments:
        image -- TensorFlow tensor representing an image.

    Returns:
        TensorFlow tensor with pixel values clipped to the range [0, 1].
    """
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def compute_content_cost(content_output, generated_output):
    """
    Computes the content cost between the content image and the generated image.

    Arguments:
        content_output -- tensor (1, n_H, n_W, n_C), hidden layer activations representing content of the image C.
        generated_output -- tensor(1, n_H, n_W, n_C), hidden layer activations representing content of the image G.

    Returns:
        J_content -- scalar representing the content cost.
    """
    a_C = content_output[-1]  # The content representation of the content image.
    a_G = generated_output[-1]  # The content representation of the generated image.

    # Retrieve dimensions from a_G
    _, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape 'a_C' and 'a_G' to match the shape (1, n_H * n_W, n_C)
    a_C_unrolled = tf.reshape(a_C, shape=[1, n_H * n_W, n_C])
    a_G_unrolled = tf.reshape(a_G, shape=[1, n_H * n_W, n_C])

    # Compute the content cost using TensorFlow
    J_content = (1. / (4. * n_H * n_W * n_C)) * tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))

    return J_content


def gram_matrix(A):
    """
    Computes the Gram matrix of a given matrix.

    Argument:
        A -- TensorFlow tensor (n_C, n_H*n_W), where n_C is the number of channels and n_H*n_W is the height times width

    Returns:
        GA -- Gram matrix of A, a TensorFlow tensor of shape (n_C, n_C).
    """
    # Compute the Gram matrix by multiplying the matrix A with its transpose
    GA = tf.linalg.matmul(A, tf.transpose(A))

    return GA


def compute_layer_style_cost(a_S, a_G):
    """
    Computes the style cost for a single layer.

    Arguments:
        a_S -- TensorFlow tensor (1, n_H, n_W, n_C), hidden layer activations representing style of the image S.
        a_G -- TensorFlow tensor (1, n_H, n_W, n_C), hidden layer activations representing style of the image G.

    Returns:
        J_style_layer -- TensorFlow tensor representing a scalar value, the style cost for the layer.
    """

    # Retrieve dimensions from a_G
    _, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape the tensors from (1, n_H, n_W, n_C) to (n_C, n_H * n_W)
    # We can’t just directly reshape the inputs into the end shape that we want; we need to do a transpose as part of
    # the process so that the “channels” dimension of the data is preserved
    a_S = tf.reshape(tf.transpose(a_S, perm=[0, 3, 1, 2]), shape=[n_C, n_H * n_W])
    a_G = tf.reshape(tf.transpose(a_G, perm=[0, 3, 1, 2]), shape=[n_C, n_H * n_W])

    # Compute the Gram matrices for both images S and G
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Compute the style cost
    J_style_layer = (1. / tf.square(2. * n_H * n_W * n_C)) * tf.reduce_sum(tf.square(tf.subtract(GS, GG)))

    return J_style_layer


def compute_style_cost(style_image_output, generated_image_output, style_layers):
    """
    Computes the overall style cost from several chosen layers.

    Arguments:
        style_image_output -- TensorFlow model output for the style image.
        generated_image_output -- TensorFlow model output for the generated image.
        style_layers -- A Python list containing:
                            - the names of the layers we would like to extract style from
                            - a coefficient for each layer

    Returns:
        J_style -- TensorFlow tensor representing a scalar value, the overall style cost.
    """
    # Initialize the overall style cost
    J_style = 0

    # Set a_S to be the hidden layer activations from the selected layers of the style image.
    a_S = style_image_output[:-1]  # Exclude the content layer activations

    # Set a_G to be the hidden layer activations from the selected layers of the generated image.
    a_G = generated_image_output[:-1]  # Exclude the content layer activations

    # Loop through each selected layer and its corresponding weight
    for i, weight in zip(range(len(a_S)), style_layers):
        # Compute the style cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S[i], a_G[i])

        # Add the weighted style cost of the current layer to the overall style cost
        J_style += weight[1] * J_style_layer

    return J_style


@tf.function()
def total_cost(J_content, J_style, alpha=10, beta=40):
    """
    Computes the total cost function combining content and style costs.

    Arguments:
        J_content -- Tensor representing the content cost.
        J_style -- Tensor representing the style cost.
        alpha -- Hyperparameter weighting the importance of the content cost (default is 10).
        beta -- Hyperparameter weighting the importance of the style cost (default is 40).

    Returns:
        J -- Tensor representing the total cost, a scalar value.

    Note:
        - Content Dominance: If preserving the original content is more important, set alpha higher.
        - Style Dominance: If transferring the style is more crucial, set beta higher.

        To emphasize the original content while incorporating the style without overwhelming it, set
        alpha (content weight) significantly higher than beta (style weight). For example, alpha = 100 and
        beta = 1. Setting alpha = 100 and beta = 10 will still prioritize content over style, but incorporate more style

        Summary: A higher alpha emphasizes preserving the original content. Conversely, a higher beta emphasizes style
    """

    # Calculate the total cost by combining content and style costs with their respective weights
    J = J_content * alpha + J_style * beta

    return J


def tensor_to_image(tensor):
    """
    Converts a TensorFlow tensor into a PIL image.

    Arguments:
        tensor -- TensorFlow tensor

    Returns:
        image -- A PIL image
    """
    # Scale the tensor values to the 0-255 range
    tensor = tensor * 255
    # Convert the tensor to a NumPy array with uint8 data type
    tensor = np.array(tensor, dtype=np.uint8)

    # If the tensor has more than 3 dimensions, remove the batch dimension
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1  # Ensure the batch dimension is 1
        tensor = tensor[0]  # Remove the batch dimension

    # Convert the NumPy array to a PIL image
    return Image.fromarray(tensor)
