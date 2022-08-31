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
		ax0.imshow(i, cmap="gray", interpolation="nearest")
		ax1.imshow(i_pred, cmap="gray", interpolation="nearest")
		plt.show()


"""
	Helper functions/classes for DBN
"""

# Just a custom except for verify_args function
class RequiredParamError(Exception): pass

def verify_args(class_rbm, args:dict) -> dict:
	""" Function to verify and filter arguments passed for a certain class
	
	Args:
	    class_rbm (class): Class to check if passed arguments are valid
	    args (dict): Passed arguments
	
	Returns:
	    dict: New dictionary with all valid arguments
	
	Raises:
	    RequiredParamError: Raised when a required parameter is missing
	"""

	# Get all parameters and split between default and required
	req_args_rbm, def_args_rbm = set(), set()
	
	for param in signature(class_rbm).parameters.values():
		if param.default == _empty: req_args_rbm.add(param.name)
		else: def_args_rbm.add(param.name)
	
	# Get all arguments passed by us
	args_keys = set(args.keys())

	# If resulting set isn't empty then we don't passed all required params
	try:
		remain_args = req_args_rbm-args_keys
		if len(remain_args) != 0:
			raise RequiredParamError

	except RequiredParamError as e:
		print("[!] You forgot to pass: ", remain_args)
		sys.exit(-1)

	# Filter any wrong parameter
	return {k:args[k] for k in args_keys.intersection(req_args_rbm.union(def_args_rbm))}
