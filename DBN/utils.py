import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import sys
from inspect import signature, _empty


# Transform ℤ: [0,255] -> ℝ: [0,1]
normalizer = tf.keras.layers.Rescaling(1.0 / 255)

# Transform ℝ: [0,1] -> ℤ: [0,255]
unormalizer = lambda x: tf.cast(tf.keras.layers.Rescaling(255)(x), tf.int32)


def get_datasets(dataset_name: str, normalize=True):
	""" Function get a dataset and prepare it to be used to efficiently on training pipeline
	
	Args:
	    dataset_name (str): Name of the dataset of tensorflow_datasets
	    normalize (bool, optional): Determine if the dataset should or shouldn't be normalized
	
	Returns:
	    tuple: Tuple of processed train, validation and test data tensors
	"""
	(img_train, img_val, img_test), ds_info = tfds.load(
	    dataset_name,
	    split=["train[:80%]", "train[80%:]", "test"],
	    as_supervised=True,
	    with_info=True,
	)

	if normalize:
		norm_train = img_train.map(lambda x, y: (normalizer(x), y)).cache().shuffle(tf.data.experimental.cardinality(img_train)).batch(32).prefetch(tf.data.AUTOTUNE)
		norm_val = img_val.map(lambda x, y: (normalizer(x), y)).cache().shuffle(tf.data.experimental.cardinality(img_val)).batch(32).prefetch(tf.data.AUTOTUNE)
		norm_test = img_test.map(lambda x, y: (normalizer(x), y)).cache().shuffle(tf.data.experimental.cardinality(img_test)).batch(32).prefetch(tf.data.AUTOTUNE)
		
		return norm_train, norm_val, norm_test, ds_info

	return img_train, img_val, img_test, ds_info

def get_pretrain_images(autoencoder, dataset_name: str):
	""" Function get a dataset and encoded it using an autoencoder

		Dataset will be always normalized!
	
	Args:
	    autoencoder (Model): Autoencoder model
	    dataset_name (str): Name of the dataset of tensorflow_datasets
	
	Returns:
	    tuple: Tuple of processed train, validation and test data tensors
	"""
	(img_train, img_val, img_test), ds_info = tfds.load(
        dataset_name,
        split=["train[:80%]", "train[80%:]", "test"],
        as_supervised=True,
        with_info=True,
    )

	r_img_train = img_train.cache().shuffle(tf.data.experimental.cardinality(img_train)).batch(32).map(lambda x, y: (autoencoder(normalizer(x)), y)).prefetch(tf.data.AUTOTUNE)
	r_img_val = img_val.cache().shuffle(tf.data.experimental.cardinality(img_train)).batch(32).map(lambda x, y: (autoencoder(normalizer(x)), y)).prefetch(tf.data.AUTOTUNE)
	r_img_test = img_test.cache().shuffle(tf.data.experimental.cardinality(img_train)).batch(32).map(lambda x, y: (autoencoder(normalizer(x)), y)).prefetch(tf.data.AUTOTUNE)

	return r_img_train, r_img_val, r_img_test

"""
	Helper functions/classes for RBMs 

"""

def show_batch_images(batch, pred, num_imgs=5, unormalize=True):
	""" Function to show original and reconstructed images side by side
		given a batch of images.
	
	Args:
	    batch (Tensor): Tensor with a batch of original images
	    pred (Tensor): Tensor with a batch of reconstructed images
	    num_imgs (int, optional): Number of images to show from batch
	"""
	for img, img_pred in zip(batch[:num_imgs], pred[:num_imgs]):

		if unormalize:
			i, i_pred = unormalizer(img).numpy(), unormalizer(img_pred).numpy()
		else:
			i, i_pred = img.numpy(), img_pred.numpy()

		fig, (ax0, ax1) = plt.subplots(1,2)
		ax0.imshow(i, interpolation="nearest")
		ax1.imshow(i_pred, interpolation="nearest")
		plt.show()


"""
	Helper functions/classes for DBN
"""

# Just a custom except for verify_args function
class RequiredParamError(Exception): pass

def get_shallow_net(in_size: tuple[int]):
	""" Function to create and compile a shallow convnet for training
	
	Args:
	    in_size (tuple[int]): Input size
	
	Returns:
	    Model: Prepared Sequential Model
	"""
	shallow = tf.keras.Sequential([
	    tf.keras.Input(shape=in_size),
	    tf.keras.layers.Conv2D(32, (3,3), activation="relu", padding="same"),
	    tf.keras.layers.MaxPooling2D(),
	    tf.keras.layers.Conv2D(64, (3,3), activation="relu", padding="same"),
	    tf.keras.layers.MaxPooling2D(),
	    tf.keras.layers.GlobalAveragePooling2D(),
	    tf.keras.layers.Flatten(),
	    tf.keras.layers.Dense(10)
	])

	shallow.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

	return shallow