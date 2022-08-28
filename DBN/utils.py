import tensorflow as tf
import matplotlib.pyplot as plt
import sys
from inspect import signature, _empty

"""
	Helper functions/classes for RBMs

"""

def show_batch_images(batch, pred, unorm_fn, num_imgs=5):
	""" Function to show original and reconstructed images side by side
		given a batch of images.
	
	Args:
	    batch (Tensor): Tensor with a batch of original images
	    pred (Tensor): Tensor with a batch of reconstructed images
	    unorm_fn (function): Function to unormalize images from [0,1] -> [0,255]
	    num_imgs (int, optional): Number of images to show from batch
	"""
	for img, img_pred in zip(batch[:num_imgs], pred[:num_imgs]):
		i, i_pred = unorm_fn(img).numpy(), unorm_fn(img_pred).numpy()

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
