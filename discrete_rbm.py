import tensorflow as tf


class RBMBernoulli(tf.keras.layers.Layer):

    """Class that represents a Discrete Restricted Boltzman Machine (Bernoulli)

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

    def __init__(self, hidden_units: int, k=1, lr=0.01):
        """
        Args:
            hidden_units (int): Number of hidden units (latent variables)
            k (int): Number of Gibbs Samplings
            lr (float): Learning rate
        """
        super(RBMBernoulli, self).__init__()

        self.h = tf.Variable(tf.zeros(shape=(hidden_units, 1)), name='h')
        self.b = tf.Variable(tf.zeros(shape=(hidden_units, 1)), name='b')

        self.k = k
        self.lr = lr

    def build(self, input_shape):
        """Receive the shape of the input

                Because we're passing an image, I start flattening their shape

        Args:
            input_shape (tuple[int]): Input shape
        """

        # input_shape = NHWC = (Batch, Height, Weight, Channels)
        self.flat_shape = input_shape[1] * input_shape[2] * input_shape[3]

        # Save the original shape to reshape input in __call__
        self.width, self.height, self.channels = (
            input_shape[2],
            input_shape[1],
            input_shape[3],
        )

        self.v = tf.Variable(tf.zeros(shape=(self.flat_shape, 1)), name='v')
        self.a = tf.Variable(tf.zeros(shape=(self.flat_shape, 1)), name='a')

        # Sampling N(μ=0, σ=0.1) to initialize weights
        self.W = tf.Variable(
            tf.random.normal(
                shape=(self.flat_shape, tf.shape(self.h)[0]), stddev=0.1
            ),
            name='W',
        )

    def call(self, inputs):
        """Receive input and transform it

                This is the place where we call other functions to calculate conditionals and reconstruct input

                NOTE: Batch are not supported for now
        Args:
            inputs (tf.Tensor): Input Tensor
        """

        # Loop over all samples in batch
        for b in range(tf.shape(inputs)[0]):

            # Reshape (remove batch dimension) to be valid during math operations
            self.v.assign(tf.reshape(inputs[b], [self.flat_shape, 1]))
            self.k_gibbs_sampling()
            self.contrastive_divergence()

        # Update reconstruction with new weights
        self.v.assign(self.v_given_h())

        # Return reconstructed input reshape to the original shape
        return tf.reshape(self.v, [self.height, self.width, self.channels])

    def k_gibbs_sampling(self):
        """Function to sample h₍₀₎ from v₍₀₎, v₍₁₎ from h₍₀₎ ... v₍ₖ₊₁₎ from h₍ₖ₎"""
        
        # Save initial input (tf.identity == np.copy)
        self.v_init = tf.identity(self.v)

        for _ in range(self.k):
            # h ~ p(h | v)
            self.h.assign(self.h_given_v(self.v))

            # v ~ p(v | h)
            self.v.assign(self.v_given_h())


    def contrastive_divergence(self):
        """Function to approximate the gradient where we have a positive(ϕ⁺) and negative(ϕ⁻) grad.

        ϕ⁻ = p(h₍ₜ₎ = 1 | v₍ₜ₎) ⋅ v₍ₜ₎
        ϕ⁺ = p(h₍₀₎ = 1 | v₍₀₎ ⋅ v₍₀₎

        ϕ⁺ - ϕ⁻ is the constrastive divergence which approximate the derivation of maximum log-likelihood
        """

        # h ~ p(h₍ₜ₎ = 1 | v₍ₜ₎)
        h_bin = self.sample_binary_prob( self.h )

        # h ~ p(h₍₀₎ = 1 | v₍₀₎)
        h_init = self.sample_binary_prob( self.h_given_v(self.v_init) )

        self.W.assign_add(
            self.lr
            * (
                tf.linalg.matmul(self.v_init, tf.transpose(h_init))
                - tf.linalg.matmul(self.v, tf.transpose(h_bin))
            )
        )
        self.a.assign_add(self.lr * (self.v_init - self.v))
        self.b.assign_add(self.lr * (h_init - h_bin))

    def v_given_h(self):
        """Function that implements the conditional probability:

            P(v | h) = σ( b + ∑ v⋅W )

        Returns:
            Tensor: Tensor of shape [tf.shape(v)[0], 1]
        """
        return tf.math.sigmoid(self.a + tf.linalg.matmul(self.W, self.h))

    def h_given_v(self, v):
        """Function that implements the conditional probability:


            P(h | v) = σ( a + ∑ h⋅W )

        Args:
            Tensor: Tensor of shape [tf.shape(v)[0], 1]

        Returns:
            Tensor: Tensor of shape [tf.shape(h)[0], 1]
        """
        return tf.math.sigmoid(
            self.b + tf.linalg.matmul(tf.transpose(self.W), v)
        )

    def sample_binary_prob(self, probs):
        """Function that transform probabilities [0,1] into a binary Tensor ( set of {0,1} )

            Here we calculate the probability of a certain element of the probs Tensor being selected
            (Uniform Dist. sampling)

            Subtracting this 2 probabilities (probs and its uniform sampling) will give a negative value
            if probability = 0 (because the given element in probs as a low probability)

            Else give a positive value, so probability = 1

            The sign function will return -1 if negative 0 if zero 1 if positive

            And the relu function will return 0 if negative OR zero 1 if positive

            With this a binary Tensor will return

        Args:
            probs (Tensor): Tensor with probabilitites [0,1]

        Returns:
            Tensor: Tensor with binary probabilities
        """
        return tf.nn.relu(tf.sign(probs - tf.random.uniform(tf.shape(probs))))
