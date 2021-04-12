from keras import backend
import numpy as np
from skimage.transform import resize, rotate


def scale_dataset(images, new_shape):
	images_list = list()
	for image in images:
		new_image1 = resize(image, new_shape,0)
		images_list.append(new_image1)
		angle1, angle2 = [np.random.randint(0,20), -np.random.randint(0,20)]
		new_image2 = rotate(image, angle1)
		new_image2 = resize(new_image2, new_shape, 0)
		images_list.append(new_image2)
		new_image3 = rotate(image, angle2)
		new_image3 = resize(new_image3, new_shape, 0)
		images_list.append(new_image3)
	return np.asarray(images_list).astype('float32')