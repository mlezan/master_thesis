from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Reshape, Conv2D, UpSampling2D, AveragePooling2D, LeakyReLU, Layer
from keras.initializers import RandomNormal
import keras
from custom_layers import *
from losses import *

def calculate_lr(input_resolution):
  """
  This function is used to calculate the learning rate of the discriminator and the generator.
  Parameters:
    input_resolution: the input resolution as an integer so for a 4x4 image this would be 4
  Returns:
    a learning rate (float)
  """
  return (0.004/input_resolution)**1.1

from keras.utils.generic_utils import get_custom_objects
get_custom_objects().update({"wasserstein_loss": wasserstein_loss})

def add_discriminator_block(old_model, n_input_layers=3):
  """
  This function adds a discriminator block to the model.
  Parameters:
    old_model: the model that you build upon
    n_input_layers: the amount of layers you define in your block, as the input has already been passed to those layers (always 3 in this code)
  Returns:
    model1: a "normal" convolutional model which will be used to train the "tuned" outputs
    model2: a convolutional model that is faded in slowly using the alpha of the WeightedArithmeticMean class
  """
  init = RandomNormal(stddev=0.02)
  const = keras.constraints.max_norm(1.0)
  in_shape = list(old_model.input.shape)
  input_shape = (in_shape[-2]*2, in_shape[-2]*2, in_shape[-1])
  lr = calculate_lr(input_shape[-2])
  in_image = Input(shape=input_shape)
  d = Conv2D(128, (1,1), padding='same', kernel_initializer=init, kernel_constraint=const)(in_image)
  d = LeakyReLU(alpha=0.2)(d)
  d = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
  d = LeakyReLU(alpha=0.2)(d)
  d = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
  d = LeakyReLU(alpha=0.2)(d)
  d = AveragePooling2D()(d)
  block_new = d
  for i in range(n_input_layers, len(old_model.layers)):
    d = old_model.layers[i](d)
  model1 = Model(in_image, d)
  model1.compile(loss=wasserstein_loss, optimizer=Adam(lr=lr, beta_1=0, beta_2=0.99, epsilon=10e-8))
  downsample = AveragePooling2D()(in_image)
  block_old = old_model.layers[1](downsample)
  block_old = old_model.layers[2](block_old)
  d = WeightedArithmeticMean()([block_old, block_new])
  # skip the input, 1x1 and activation for the old model
  for i in range(n_input_layers, len(old_model.layers)):
    d = old_model.layers[i](d)
  model2 = Model(in_image, d)
  model2.compile(loss=wasserstein_loss, optimizer=Adam(lr=lr, beta_1=0, beta_2=0.99, epsilon=10e-8))
  return [model1, model2]

def define_discriminator(n_blocks, input_shape=(4,4,3)):
  """
  Function that builds the discriminator using discriminator blocks
  Parameters:
    n_blocks: the amount of blocks you want to make, which depends on the image resolution you want to generate
    input_shape: input shape of your images. For grayscale this would be (4,4,1). If you want to start generating
      at a higher resolution you could increment it to (8,8,3) / (8,8,1)
  Returns:
    A list of lists containing all the faded and tuned models (nested) at different resolutions.
  """
  init = RandomNormal(stddev=0.02)
  const = keras.constraints.max_norm(1.0)
  model_list = list()
  in_image = Input(shape=input_shape)
  d = Conv2D(128, (1,1), padding='same', kernel_initializer=init, kernel_constraint=const)(in_image)
  d = LeakyReLU(alpha=0.2)(d)
  d = MinibatchStdev()(d)
  d = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
  d = LeakyReLU(alpha=0.2)(d)
  d = Conv2D(128, (4,4), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
  d = LeakyReLU(alpha=0.2)(d)
  d = Flatten()(d)
  out_class = Dense(1)(d)
  model = Model(in_image, out_class)
  model.compile(loss=wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
  model_list.append([model, model])
  for i in range(1, n_blocks):
    old_model = model_list[i - 1][0]
    models = add_discriminator_block(old_model)
    model_list.append(models)
  return model_list

def add_generator_block(old_model):
  """
  Function to add a generator block to your generator.
  Parameters:
    old_model: the model you will add this new block to
  Returns:
    model1: a "normal" convolutional model which will be used to train the "tuned" outputs
    model2: a convolutional model that is faded in slowly using the alpha of the WeightedArithmeticMean class
  """
  init = RandomNormal(stddev=0.02)
  const = keras.constraints.max_norm(1.0)
  block_end = old_model.layers[-2].output
  upsampling = UpSampling2D()(block_end)
  g = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(upsampling)
  g = PixelNormalization()(g)
  g = LeakyReLU(alpha=0.2)(g)
  g = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
  g = PixelNormalization()(g)
  g = LeakyReLU(alpha=0.2)(g)
  out_image = Conv2D(3, (1,1), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
  model1 = Model(old_model.input, out_image)
  out_old = old_model.layers[-1]
  out_image2 = out_old(upsampling)
  merged = WeightedArithmeticMean()([out_image2, out_image])
  model2 = Model(old_model.input, merged)
  return [model1, model2]

def define_generator(latent_dim, n_blocks, in_dim=4):
  """
  Function that builds the generator using generator blocks
  Parameters:
    latent_dim: the size of the input noise vector
    n_blocks: the amount of blocks you need, which depends on the resolution of images you want to generate
    in_dim: the starting resolution of your pggan
  """
  init = RandomNormal(stddev=0.02)
  const = keras.constraints.max_norm(1.0)
  model_list = list()
  in_latent = Input(shape=(latent_dim,))
  g  = Dense(128 * in_dim * in_dim, kernel_initializer=init, kernel_constraint=const)(in_latent)
  g = Reshape((in_dim, in_dim, 128))(g)
  g = Conv2D(128, (4,4), padding='same', kernel_initializer=init, kernel_constraint=const)(g) ## of (3,3)
  g = PixelNormalization()(g)
  g = LeakyReLU(alpha=0.2)(g)
  g = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
  g = PixelNormalization()(g)
  g = LeakyReLU(alpha=0.2)(g)
  out_image = Conv2D(3, (1,1), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
  model = Model(in_latent, out_image)
  model_list.append([model, model])
  for i in range(1, n_blocks):
    old_model = model_list[i - 1][0]
    models = add_generator_block(old_model)
    model_list.append(models)
  return model_list

def define_composite(discriminators, generators):
  """
  The composite is used to train the generators via the discriminators
  Parameters:
    discriminators: the discriminators
    generators: the generators
  Returns:
    A list of lists containing all the faded and tuned models (nested) at different resolutions.
  """
  model_list = list()
  for i in range(len(discriminators)):
    g_models, d_models = generators[i], discriminators[i]
    lr = calculate_lr(g_models[0].output.shape[-2])
    d_models[0].trainable = False
    model1 = Sequential()
    model1.add(g_models[0])
    model1.add(d_models[0])
    model1.compile(loss=wasserstein_loss, optimizer=Adam(lr=lr, beta_1=0, beta_2=0.99, epsilon=10e-8))
    d_models[1].trainable = False
    model2 = Sequential()
    model2.add(g_models[1])
    model2.add(d_models[1])
    model2.compile(loss=wasserstein_loss, optimizer=Adam(lr=lr, beta_1=0, beta_2=0.99, epsilon=10e-8))
    model_list.append([model1, model2])
  return model_list