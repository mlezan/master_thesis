from keras import backend
import matplotlib.pyplot as plt
import numpy as np
from custom_layers import WeightedArithmeticMean

def update_fadein(models, step, n_steps):
	"""
	This function sets the alpha for each model
	Parameters:
		models: the two models that need to be combined using weighted arithmetic mean, i.e. the upsampled / downsampled version and the convolutional net
		step: the step size that you're at at that point
		n_steps: the total amount of steps
		these two values will determine alpha
	Returns:
		nothing, it updates the alpha of the layer
	"""
	alpha = step / float(n_steps - 1)
	for model in models:
		for layer in model.layers:
			if isinstance(layer, WeightedArithmeticMean):
				backend.set_value(layer.alpha, alpha)

def load_real_samples(filename):
	"""
	This method loads the data from a file, transforms it to float and scales it from [0,255] to [-1,1]
	Parameters:
		filename: the filename which loads your files, this would be an .npz file of your images
	Returns:
		rescaled, float array containing your image data
	"""
	data = np.load(filename)
	X = data['arr_0']
	X = X.astype('float32')
	X = (X - 127.5) / 127.5
	return X

def generate_real_samples(dataset, n_samples):
	"""
	This method retrieves real samples from your dataset
	Parameters:
		dataset: the rescaled dataset, which is a numpy array
		n_samples: the amount of samples you want to generate
	Returns:
		X: a number of random samples from your dataset
		y: a number of ones matching the amount of samples, indicating that these samples are "real"
	"""
	ix = np.random.randint(0, dataset.shape[0], n_samples)
	X = dataset[ix]
	y = np.ones((n_samples, 1))
	return X, y

def generate_latent_points(latent_dim, n_samples):
	"""
	Generates a latent vector
	Parameters:
		latent_dim: the size of your latent vector, in our code 100
		n_samples: the amount of samples you want to generate
	Returns:
		a random vector of random numbers that can be used to generate images of size (n_samples, latent_dim)
	"""
	x_input = np.random.randn(latent_dim * n_samples)
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

def generate_fake_samples(generator, latent_dim, n_samples):
	"""
	Generates fake samples
	Parameters:
		generator: the generator you want to use to generate samples with
		latent_dim: the size of your latent vector, in our code 100
		n_samples: the amount of samples you want to generate
	Returns:
		X: the generated images
		y: a number of -1's matching the amount of samples, indicating that these samples are "fake"
	"""
	x_input = generate_latent_points(latent_dim, n_samples)
	X = generator.predict(x_input)
	y = -np.ones((n_samples, 1))
	return X, y


def summarize_performance(status, g_model, d_model, gan_model, latent_dim, n_samples=25, output_folder="results4"):
	"""
	Generate samples, save them as a plot and save the model
	Parameters:
		status: faded or tuned
		g_model: the generator
		d_model: the discriminator
		gan_model: the composite that trains the generator
		latent_dim: the size of the latent vector
		n_samples: the amount of samples you want to generate and store. Needs to be a value that you can take the square root of
		output_folder: the folder in which you want to store your results. Make sure this folder is already made
	Returns:
		Nothing, but saves the plot, the losses and the model
	"""
	gen_shape = g_model.output_shape
	name = '%03dx%03d-%s' % (gen_shape[1], gen_shape[2], status)
	X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
	X = (X - X.min()) / (X.max() - X.min())
	square = int(math.sqrt(n_samples))
	for i in range(n_samples):
		plt.subplot(square, square, 1 + i)
		plt.axis('off')
		plt.imshow(X[i])
	filename1 = output_folder+'/plot_%s.png' % (name)
	plt.savefig(filename1)
	plt.close()
	filename2 = output_folder+'/g_model_%s' % (name)
	g_model.save(filename2)
	filename3 = output_folder+'/d_model_%s' % (name)
	d_model.save(filename3)
	filename4 = output_folder+'/gan_model_%s' % (name)
	gan_model.save(filename4)
	print('>Saved: %s and %s and %s and %s' % (filename1, filename2, filename3, filename4))