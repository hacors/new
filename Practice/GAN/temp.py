import os
from IPython import display
import time
import PIL
import numpy as np
import matplotlib.pyplot as plt
import imageio
import glob
import tensorflow as tf
Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
try:
    os.chdir(os.path.join(os.getcwd(), 'Practice\GAN\refers\tensorflow\contrib\eager\python\examples\generative_examples'))
    print(os.getcwd())
except:
    pass

# **Copyright 2018 The TensorFlow Authors**.
#
# Licensed under the Apache License, Version 2.0 (the "License").
#
# # Generating Handwritten Digits with DCGAN
#
# <table class="tfo-notebook-buttons" align="left"><td>
# <a target="_blank"  href="https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/generative_examples/dcgan.ipynb">
#     <img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
# </td><td>
# <a target="_blank"  href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/eager/python/examples/generative_examples/dcgan.ipynb"><img width=32px src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a></td></table>

# This tutorial demonstrates how to generate images of handwritten digits using a Deep Convolutional Generative Adversarial Network ([DCGAN](https://arxiv.org/pdf/1511.06434.pdf)). The code is written in [tf.keras](https://www.tensorflow.org/programmers_guide/keras) with [eager execution](https://www.tensorflow.org/programmers_guide/eager) enabled.

# >[Generating Handwritten Digits with DCGAN](#scrollTo=0TD5ZrvEMbhZ)
#
# >>[What are GANs?](#scrollTo=2MbKJY38Puy9)
#
# >>>[Import TensorFlow and enable eager execution](#scrollTo=e1_Y75QXJS6h)
#
# >>>[Load the dataset](#scrollTo=iYn4MdZnKCey)
#
# >>>[Use tf.data to create batches and shuffle the dataset](#scrollTo=PIGN6ouoQxt3)
#
# >>[Create the models](#scrollTo=THY-sZMiQ4UV)
#
# >>>[The Generator Model](#scrollTo=-tEyxE-GMC48)
#
# >>>[The Discriminator model](#scrollTo=D0IKnaCtg6WE)
#
# >>[Define the loss functions and the optimizer](#scrollTo=0FMYgY_mPfTi)
#
# >>>[Generator loss](#scrollTo=Jd-3GCUEiKtv)
#
# >>>[Discriminator loss](#scrollTo=PKY_iPSPNWoj)
#
# >>[Set up GANs for Training](#scrollTo=Rw1fkAczTQYh)
#
# >>[Train the GANs](#scrollTo=dZrd4CdjR-Fp)
#
# >>[Generated images](#scrollTo=P4M_vIbUi7c0)
#
# >>[Learn more about GANs](#scrollTo=k6qC-SbjK0yW)
#
#

# ## What are GANs?
# GANs, or [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661), are a framework for estimating generative models. Two models are trained simultaneously by an adversarial process: a Generator, which is responsible for generating data (say, images), and a Discriminator, which is responsible for estimating the probability that an image was drawn from the training data (the image is real), or was produced by the Generator (the image is fake). During training, the Generator becomes progressively better at generating images, until the Discriminator is no longer able to distinguish real images from fake.
#
# ![alt text](https://github.com/margaretmz/tensorflow/blob/margaret-dcgan/tensorflow/contrib/eager/python/examples/generative_examples/gans_diagram.png?raw=1)
#
# We will demonstrate this process end-to-end on MNIST. Below is an animation that shows a series of images produced by the Generator as it was trained for 50 epochs. Overtime, the generated images become increasingly difficult to distinguish from the training set.
#
# To learn more about GANs, we recommend MIT's [Intro to Deep Learning](http://introtodeeplearning.com/) course, which includes a lecture on Deep Generative Models ([video](https://youtu.be/JVb54xhEw6Y) | [slides](http://introtodeeplearning.com/materials/2018_6S191_Lecture4.pdf)). Now, let's head to the code!
#
# ![sample output](https://tensorflow.org/images/gan/dcgan.gif)


# Install imgeio in order to generate an animated gif showing the image generating process
get_ipython().system('pip install imageio')


# ### Import TensorFlow and enable eager execution


tf.enable_eager_execution()


# ### Load the dataset
#
# We are going to use the MNIST dataset to train the generator and the discriminator. The generator will generate handwritten digits resembling the MNIST data.


(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()


train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]


BUFFER_SIZE = 60000
BATCH_SIZE = 256


# ### Use tf.data to create batches and shuffle the dataset


train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


# ## Create the models
#
# We will use tf.keras [Sequential API](https://www.tensorflow.org/guide/keras#sequential_model) to define the generator and discriminator models.

# ### The Generator Model
#
# The generator is responsible for creating convincing images that are good enough to fool the discriminator. The network architecture for the generator consists of [Conv2DTranspose](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2DTranspose) (Upsampling) layers. We start with a fully connected layer and upsample the image two times in order to reach the desired image size of 28x28x1. We increase the width and height, and reduce the depth as we move through the layers in the network. We use [Leaky ReLU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LeakyReLU) activation for each layer except for the last one where we use a tanh activation.


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model


# ### The Discriminator model
#
# The discriminator is responsible for distinguishing fake images from real images. It's similar to a regular CNN-based image classifier.


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))

    return model


generator = make_generator_model()
discriminator = make_discriminator_model()


# ## Define the loss functions and the optimizer
#
# Let's define the loss functions and the optimizers for the generator and the discriminator.
#

# ### Generator loss
# The generator loss is a sigmoid cross entropy loss of the generated images and an array of ones, since the generator is trying to generate fake images that resemble the real images.


def generator_loss(generated_output):
    return tf.losses.sigmoid_cross_entropy(tf.ones_like(generated_output), generated_output)


# ### Discriminator loss
#
# The discriminator loss function takes two inputs: real images, and generated images. Here is how to calculate the discriminator loss:
# 1. Calculate real_loss which is a sigmoid cross entropy loss of the real images and an array of ones (since these are the real images).
# 2. Calculate generated_loss which is a sigmoid cross entropy loss of the generated images and an array of zeros (since these are the fake images).
# 3. Calculate the total_loss as the sum of real_loss and generated_loss.


def discriminator_loss(real_output, generated_output):
    # [1,1,...,1] with real output since it is true and we want our generated examples to look like it
    real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(real_output), logits=real_output)

    # [0,0,...,0] with generated images since they are fake
    generated_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(generated_output), logits=generated_output)

    total_loss = real_loss + generated_loss

    return total_loss


# The discriminator and the generator optimizers are different since we will train two networks separately.


generator_optimizer = tf.train.AdamOptimizer(1e-4)
discriminator_optimizer = tf.train.AdamOptimizer(1e-4)


# **Checkpoints (Object-based saving)**


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


# ## Set up GANs for Training
#
#

# Now it's time to put together the generator and discriminator to set up the Generative Adversarial Networks, as you see in the diagam at the beginning of the tutorial.

# **Define training parameters**


EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

# We'll re-use this random vector used to seed the generator so
# it will be easier to see the improvement over time.
random_vector_for_generation = tf.random_normal([num_examples_to_generate,
                                                 noise_dim])


# **Define training method**
#
# We start by iterating over the dataset. The generator is given a random vector as an input which is processed to  output an image looking like a handwritten digit. The discriminator is then shown the real MNIST images as well as the generated images.
#
# Next, we calculate the generator and the discriminator loss. Then, we calculate the gradients of loss with respect to both the generator and the discriminator variables.


def train_step(images):
    # generating noise from a normal distribution
    noise = tf.random_normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        generated_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(generated_output)
        disc_loss = discriminator_loss(real_output, generated_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.variables))


#
# This model takes about ~30 seconds per epoch to train on a single Tesla K80 on Colab, as of October 2018.
#
# Eager execution can be slower than executing the equivalent graph as it can't benefit from whole-program optimizations on the graph, and also incurs overheads of interpreting Python code. By using [tf.contrib.eager.defun](https://www.tensorflow.org/api_docs/python/tf/contrib/eager/defun) to create graph functions, we get a ~20 secs/epoch performance boost (from ~50 secs/epoch down to ~30 secs/epoch). This way we get the best of both eager execution (easier for debugging) and graph mode (better performance).


train_step = tf.contrib.eager.defun(train_step)


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for images in dataset:
            train_step(images)

        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                 epoch + 1,
                                 random_vector_for_generation)

        # saving (checkpoint) the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time taken for epoch {} is {} sec'.format(epoch + 1,
                                                         time.time()-start))
    # generating after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epochs,
                             random_vector_for_generation)


# **Generate and save images**
#
#


def generate_and_save_images(model, epoch, test_input):
    # make sure the training parameter is set to False because we
    # don't want to train the batchnorm layer when doing inference.
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


# ## Train the GANs
# We will call the train() method defined above to train the generator and discriminator simultaneously. Note, training GANs can be tricky. It's important that the generator and discriminator do not overpower each other (e.g., that they train at a similar rate).
#
# At the beginning of the training, the generated images look like random noise. As training progresses, you can see the generated digits look increasingly real. After 50 epochs, they look very much like the MNIST digits.


get_ipython().run_cell_magic('time', '', 'train(train_dataset, EPOCHS)')


# **Restore the latest checkpoint**


# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


# ## Generated images
#

#
# After training, its time to generate some images!
# The last step is to plot the generated images and voila!
#


# Display a single image using the epoch number
def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))


display_image(EPOCHS)


# **Generate a GIF of all the saved images**
#
# We will use imageio to create an animated gif using all the images saved during training.


with imageio.get_writer('dcgan.gif', mode='I') as writer:
    filenames = glob.glob('image*.png')
    filenames = sorted(filenames)
    last = -1
    for i, filename in enumerate(filenames):
        frame = 2*(i**0.5)
        if round(frame) > round(last):
            last = frame
        else:
            continue
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)

# this is a hack to display the gif inside the notebook
os.system('cp dcgan.gif dcgan.gif.png')


# Display the animated gif with all the mages generated during the training of GANs.


display.Image(filename="dcgan.gif.png")


# **Download the animated gif**
#
# Uncomment the code below to download an animated gif from Colab.


#from google.colab import files
# files.download('dcgan.gif')


# ## Learn more about GANs
#

# We hope this tutorial was helpful! As a next step, you might like to experiment with a different dataset, for example the Large-scale Celeb Faces Attributes (CelebA) dataset [available on Kaggle](https://www.kaggle.com/jessicali9530/celeba-dataset/home).
#
# To learn more about GANs:
#
# * Check out MIT's lecture (linked above), or [this](http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture12.pdf) lecture form Stanford's CS231n.
#
# * We also recommend the [CVPR 2018 Tutorial on GANs](https://sites.google.com/view/cvpr2018tutorialongans/), and the [NIPS 2016 Tutorial: Generative Adversarial Networks](https://arxiv.org/abs/1701.00160).
#
