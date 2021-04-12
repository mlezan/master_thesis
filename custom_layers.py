from keras.layers import Add, Layer
from keras import backend

class WeightedArithmeticMean(Add):
  """
  This class takes the weighted arithmetic mean of an upsampled/downsampled image and the convolutional model
  The "weighted" part is determined by alpha, starting with an alpha of 0 meaning that 100% of the weighted
  average is determined by the upsampling(NN)/downsampling and end with an alpha of 1 meaning that 100% 
  of the weighted average is determined by the CNN
  """
  def __init__(self, alpha=0.0, **kwargs):
    super(WeightedArithmeticMean, self).__init__(**kwargs)
    self.alpha = backend.variable(alpha, name='ws_alpha')

  def _merge_function(self, inputs):
    """
    Calculates weighted average. Overridden from _merge_function in parent class Add.
    """
    assert (len(inputs) == 2)
    output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
    return output

  def get_config(self):
    """
    Necessary for serialization. This implementation is a workaround because alpha isn't passed (backend.variable is not serializable)
    but it doesn't matter because when we serialize the model at the end of training alpha will be 1 anyways.
    """
    config = super(WeightedArithmeticMean, self).get_config().copy()
    return config

class PixelNormalization(Layer):
  """
  Pixel Normalization has been proposed to combat gradient explosion. It works by normalizing the feature vector in each pixel.
  It functions as a replacement for Batch Normalization.
  """
  def __init__(self, **kwargs):
    super(PixelNormalization, self).__init__(**kwargs)

  def call(self, inputs):
    """
    Call is the inherited methods from the Layer parent class that is called when the layer is initialized. PixelNormalization is
    calculated as a variation on the local response normalization, see formula in our report.
    """
    values = inputs**2.0
    mean_values = backend.mean(values, axis=-1, keepdims=True)
    mean_values += 1.0e-8
    l2 = backend.sqrt(mean_values)
    normalized = inputs / l2
    return normalized

class MinibatchStdev(Layer):
  """
  This is a sort of normalization layer that encourages minibatches of generated and training images to show similar statistics. It is
  added as a layer towards the end of the discriminator. I'm saying sort of normalization because it is not actually changing the
  actual tensor values but rather it is concatenated to them as just another feature map.
  """
  def __init__(self, **kwargs):
    super(MinibatchStdev, self).__init__(**kwargs)

  def call(self, inputs):
    """
      Call is again the inherited methods from the Layer parent class that is called when the layer is initialized. The Minibatch
      StdDev is calculated by computing the std dev for each feature in each spatial location over the minibatch. 
      Then these estimates are averaged over all features and spatial locations to arrive at a single value. 
      The value is then replicated and concatenated to all spatial locations and over the minibatch, 
      yielding one additional (constant) feature map.
    """
    mean = backend.mean(inputs, axis=0, keepdims=True)
    squ_diffs = backend.square(inputs - mean)
    mean_sq_diff = backend.mean(squ_diffs, axis=0, keepdims=True) ### = variance
    mean_sq_diff += 1e-8 # add a small value to avoid a blow-up when we calculate stdev
    stdev = backend.sqrt(mean_sq_diff)
    mean_pix = backend.mean(stdev, keepdims=True)
    # this needs to be upscaled to the size of one input feature map for each sample
    shape = backend.shape(inputs)
    output = backend.tile(mean_pix, (shape[0], shape[1], shape[2], 1))
    # and finally concatenated with the output
    combined = backend.concatenate([inputs, output], axis=-1)
    return combined
