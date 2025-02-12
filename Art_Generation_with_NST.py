"""
Art Generation with Neural Style Transfer (NST):
The steps implemented in this code are as follows:

1. Visualize the content and style image
2. Select model for transfer learning (e.g., the VGG19)
3. Resize the content and style image to desired input shape
3. Randomly initialize the image to be generated
4. Build and train the model
    1. Build the model
    2. Compute the costs: content, style, and total cost
    3. Define the optimizer and learning rate
    7. Train the model
5. Test with your picture
"""

from utils_nst import *
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
import pprint


# VISUALIZE CONTENT AND STYLE IMAGE
# Content image
content_image = Image.open(os.path.join(os.getcwd(), 'images', 'louvre.jpg'))
imshow(content_image)
plt.show()

# Style image
style_image = Image.open(os.path.join(os.getcwd(), 'images', 'style_claude_monet.jpg'))
imshow(style_image)
plt.show()


# SELECT VGG19: THE PRETRAINED MODEL FOR TRANSFER LEARNING
tf.random.set_seed(42)
img_size = 400
vgg = tf.keras.applications.VGG19(include_top=False,
                                  input_shape=(img_size, img_size, 3),
                                  weights='imagenet')

vgg.trainable = False
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(vgg)

# A look at the layers
for layer in vgg.layers:
    print(layer.name)

# A look at the output of a layer 'block5_conv4,' that we'll later define as the content layer to represent the image.
print(vgg.get_layer('block5_conv4').output)  # Output: shape=(None, 25, 25, 512)


# RESIZE CONTENT AND STYLE IMAGES
content_image = np.array(content_image.resize((img_size, img_size)))
mod_shape = (1,) + content_image.shape  # Output: (1, 400, 400, 3)
content_image = tf.constant(np.reshape(content_image, mod_shape))

print(content_image.shape)  # Output: (1, 400, 400, 3)
imshow(content_image[0])
plt.show()

style_image = np.array(style_image.resize((img_size, img_size)))
style_image = tf.constant(np.reshape(style_image, mod_shape))
imshow(style_image[0])
plt.show()


# RANDOMLY INITIALIZE THE IMAGE TO BE GENERATED
generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
noise = tf.random.uniform(shape=tf.shape(generated_image), minval=-0.25, maxval=0.25)  # Noise data
generated_image = tf.add(generated_image, noise)  # Adding the noise to the image
generated_image = clip_values_0_1(generated_image)  # Clips tensor values  to be between 0 and 1

print(generated_image.shape)
imshow(generated_image.numpy()[0])
plt.show()


# BUILD MODEL FROM PRE-TRAINED VGG19
def get_layers_outputs(base_model, layer_names):
    """
    Creates a model that returns a list of intermediate layer outputs from the given base model.

    Arguments:
        base_model -- Pre-trained model.
        layer_names -- List of strings, names of the layers whose outputs are desired.

    Returns:
        model -- A new model that outputs the intermediate values of the specified layers.
    """
    # Extract the outputs of the specified layers
    outputs = [base_model.get_layer(layer_info[0]).output for layer_info in layer_names]

    # Create a new model that takes the same input as the original model and outputs the specified layers
    model = Model(inputs=[base_model.input], outputs=outputs)

    return model


# Select the layers to represent the style of the image and assign style costs
# Adding weights for each layer to balance their influence on the final style output
Selected_Style_Layers = [('block1_conv1', 0.2),
                         ('block2_conv1', 0.2),
                         ('block3_conv1', 0.2),
                         ('block4_conv1', 0.2),
                         ('block5_conv1', 0.2)]

content_layer = [('block5_conv4', 1)]
vgg_model_outputs = get_layers_outputs(vgg, Selected_Style_Layers + content_layer)

# Save the encoding of the content and style images in a separate variable
content_target = vgg_model_outputs(content_image)  # Content encoder
style_targets = vgg_model_outputs(style_image)     # Style encoder


# TRAIN THE MODEL
# a_C is the content image encoding for layer "block5_conv4"
preprocessed_content = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
a_C = vgg_model_outputs(preprocessed_content)

# a_S is the style image encoding for layers specified in Selected_Style_Layers.
preprocessed_style = tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
a_S = vgg_model_outputs(preprocessed_style)

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)


@tf.function()
def train_step(gen_image):
    """
    Performs a single training step for updating the generated image to minimize the style and content costs.

    Arguments:
        gen_image -- TensorFlow variable representing the generated image.

    Returns:
        J -- Tensor representing the total cost after the training step.
    """
    with tf.GradientTape() as tape:
        # a_G as the vgg_model_outputs for the current generated image
        a_G = vgg_model_outputs(gen_image)

        # Compute the style cost
        J_style = compute_style_cost(a_S, a_G, style_layers=Selected_Style_Layers)

        # Compute the content cost
        J_content = compute_content_cost(a_C, a_G)

        # Compute the total cost
        J = total_cost(J_content, J_style, alpha=10, beta=40)  # i.e., 1:4 ratio
        # J = total_cost(J_content, J_style, alpha=1, beta=10000)  # Original paper scale is 1:1000 and 1:10000

    # Compute gradients of the total cost with respect to the generated image
    grad = tape.gradient(J, gen_image)

    # Apply gradients to the optimizer to update the generated image
    optimizer.apply_gradients([(grad, gen_image)])

    # Clip the pixel values of the updated generated image to the range [0, 1]
    clipped_image = clip_values_0_1(gen_image)
    generated_image.assign(clipped_image)  # Update the generated image with the clipped values

    return J


generated_image = tf.Variable(generated_image)  # Initialized as a tf.Variable since it needs to be trainable
epochs = 2501
J_hist = []

for i in range(epochs):
    cost = train_step(generated_image)
    J_hist.append(cost)

    if i % 250 == 0:
        print(f"Epoch {i} ")
        plt.plot(np.arange(len(J_hist)), np.array(J_hist))
        plt.show()

    if i % 250 == 0:
        image = tensor_to_image(generated_image)
        imshow(image)

        output_dir = os.path.join(os.getcwd(), 'output')    # Define the base directory for output images
        os.makedirs(output_dir, exist_ok=True)              # Ensure the output directory exists
        image.save(os.path.join(output_dir, f"image_{i}.jpg"))  # Save the image using a dynamic path
        plt.show()

# See all three images in a row
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

images = [content_image[0], style_image[0], np.array(generated_image[0])]
titles = ['Content image', 'Style image', 'Generated image']

for ax, img, title in zip(axes, images, titles):
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(title)

plt.show()


