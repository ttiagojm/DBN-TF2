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
	def __init__(self, rbms: list):
		"""
		Args:
		    rbm_type (str): Type of RBM (CRBM or RBM)
		    rbm_num (int): Number of RBM layers
		    decrease_val (int): Value to decrease the size of each hidden layer
		    rbm_params (dict): Parameters for RBM class
		"""
		super(DBN, self).__init__()

		if not rbms:
			self.rbms = list()
		else:
			self.rbms = rbms


	def call(self, inputs, training=False):
		""" Function to train DBN
		
		Args:
		    inputs (Tensor): Input Tensor
		
		Returns:
		    Tensor: Latent/Hidden layer Tensor
		"""
		for rbm in self.rbms:
			rbm.training = training
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
			rbm.training = False
			x = rbm(x)
		
		# Here x will be h, because the reconstruction of all RBMs (except first) is the h of the previous RBM
		for rbm in reversed(self.rbms):
			x = rbm.v_given_h(x)
		
		return x

	def fit(self, inputs, epochs=1):
		for epoch in range(epochs):
			print(f"#### Epoch {epoch+1} ####")
			for i, batch in enumerate(inputs):
				print(f"{i+1} batch")
				self.call(batch[0], training=True)


		# After training, freeze the model
		for rbm in self.rbms:
			rbm.training = False