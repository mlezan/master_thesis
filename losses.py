from keras import backend
import numpy as np

def wasserstein_loss(y_true, y_pred):
	"""
	Wasserstein loss without gradient penalty.
	In order to be able to load the model later on, you'll have store it as a custom object (see below)
	Credits for fix: https://github.com/keras-team/keras/issues/5916
	Parameters:
		y_true: the real output value (in this case -1 in case of fake, 1 in case of real)
		y_pred: the predicted output value
	Returns:
		the wasserstein loss
	"""
	return backend.mean(y_true * y_pred)

def loss_gradient_penalty(y_true, y_pred, averaged_samples, gradient_penalty_weight=10):
	"""
	gradient penalty implementation. 
	Source: https://github.com/PacktPublishing/Hands-On-Generative-Adversarial-Networks-with-Keras/blob/ac1b8af1678af352e7e9efdcc6a3e829c6aed294/Chapter09/train_wgan_gp.py
	Parameters:
		y_true: the real output value (in this case -1 in case of fake, 1 in case of real)
		y_pred: the predicted output value
		averaged_samples: the weighted arithmetic mean of the two models
		gradient_penalty_weight: the weight of the regularization term
	Returns:
		the gradient penalty
	"""
	gradients = backend.gradients(y_pred, averaged_samples)[0]
	gradients_sqr = backend.square(gradients)
	gradients_sqr_sum = backend.sum(gradients_sqr,axis=np.arange(1, len(gradients_sqr.shape)))
	gradient_l2_norm = backend.sqrt(gradients_sqr_sum)
	gradient_penalty = gradient_penalty_weight * backend.square(gradient_l2_norm - 1)
	return backend.mean(gradient_penalty)