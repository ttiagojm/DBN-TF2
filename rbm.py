import tensorflow as tf
import tensorflow_datasets as tfds

class RBMBernoulli(tf.keras.layers.Layer):

	""" Class that represents a Discrete Restricted Boltzman Machine (Bernoulli)

		RBMs are represented by an Energy Function -> E(v,h) = -vᵀ⋅W⋅h - aᵀ⋅v - bᵀ⋅h
		Where v is input vector ; W is weight matrix ; h is latent vector ; a is v bias vector and b is h bias vector

		The joint probability is the exponential of -E(v,h) divided by the partition function (to transform energies in 
		probabilities)
		
		Partition functions are computationally expensive to calculate (lots of sums) and derivative (in case of 
		deriving the max log-likelihood)

		To prevent that conditionals are used to:
			* Map Input (v) into latent space (h)
			* Reconstruct Input (v) using latent space (h)

		Gibbs Sampling does the job and Contrastive Divergence will be used to approximate gradients.

		P(v=1 | h) = σ( b + ∑ v⋅W )
		P(h=1 | v) = σ( a + ∑ h⋅W )

		Sigmoids (σ) will be used as activations, because v and h should be binary random variables.
		References: (Keyvanrad, Mohammad Ali and Homayounpour, Mohammad Mehdi, 2014) | 
					(Fischer, Asja and Igel, Christian, 2012)
	"""

	def __init__(self, hidden_units: int):
		"""
		Args:
		    hidden_units (int): Number of hidden units (latent variables)
		"""
		super(RBMBernoulli, self).__init__()
		self.hidden_units = hidden_units

	def build (self, input_shape):
		""" Receive the shape of the input

			Because we're passing an image, I start flattening their shape
		
		Args:
		    input_shape (tuple[int]): Input shape
		"""
		#input_shape = NHWC = (Batch, Height, Weight, Channels)
		flat_shape = input_shape[1] * input_shape[2] * input_shape[3]

	def call(self, inputs):
		""" Receive input and transform it

			This is the place where we call other functions to calculate conditionals and reconstruct input
		
		Args:
		    inputs (tf.Tensor): Input Tensor
		"""
		pass



## [!] Just for testing

# Get one image from cifar-10
(img, ), ds_info = tfds.load(
    "cifar10",
    split=["train[:1]"],
    as_supervised=True,
    with_info=True,
)

in_size = ds_info.features["image"].shape

normalizer = tf.keras.layers.Rescaling(1.0 / 255)

# Ignore y (image class), only need the image
norm_img = img.map(lambda x, y: normalizer(x))

model = tf.keras.Sequential([
	tf.keras.Input(shape=in_size),
	RBMBernoulli(hidden_units=28)
])