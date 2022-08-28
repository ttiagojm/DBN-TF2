from crbm import RBMConv
from rbm import RBMBernoulli
from utils import verify_args
import tensorflow as tf
from utils import show_batch_images


class DBN(tf.keras.layers.Layer):
	"""Class that represents Deep Belief Network

		This network samples from joint probabilities over all RBMs.
		We can do that sampling from P(h | v) of each RBM.
		The input of each RBM is the h from the previous RBM.

		RBMs are trained separately.
	"""
	def __init__(self, rbm_type:str, rbm_num:int, decrease_val: int, rbm_params:dict):
		"""
		Args:
		    rbm_type (str): Type of RBM (CRBM or RBM)
		    rbm_num (int): Number of RBM layers
		    decrease_val (int): Value to decrease the size of each hidden layer
		    rbm_params (dict): Parameters for RBM class
		"""
		super(DBN, self).__init__()

		# RBMBernoulli is the name of RBM class
		class_rbm = RBMBernoulli if rbm_type == "RBMBernoulli" else RBMConv

		# Verify and filter passed arguments
		rbm_params = verify_args(class_rbm, rbm_params)

		# Generate all RBMs, always decreasing the size of hidden layer
		self.rbms = list()

		for i, _ in enumerate(range(rbm_num)):
			if i > 0: rbm_params["hidden_units"] -= decrease_val
			self.rbms.append(class_rbm(**rbm_params))


	def call(self, inputs):
		""" Function to train DBN
		
		Args:
		    inputs (Tensor): Input Tensor
		
		Returns:
		    Tensor: Latent/Hidden layer Tensor
		"""
		for rbm in self.rbms:
			inputs = rbm(inputs)

		return inputs

	def get_reconstruction(self, x):
		"""Function to return the reconstruction of x based on W, a and b learned previously

        Args:
            x (Tensor): Input Tensor

        Returns:
            Tensor: Reconstructed Input Tensor
        """

		## Feed forward with Gibbs Sampling
		for rbm in self.rbms:
			x = rbm(x, train=False)
		
		# Here x will be h, because the reconstruction of all RBMs (except first) is the h of the previous RBM
		for rbm in reversed(self.rbms):
			x = rbm.v_given_h(x)
		
		return x