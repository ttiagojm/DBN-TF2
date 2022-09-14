import tensorflow as tf

class RBMBernoulli(tf.keras.layers.Layer):

    """Class that represents a Discrete Restricted Boltzman Machine (Bernoulli)

    RBMs are represented by an Energy Function -> E(v,h) = -vᵀ⋅W⋅h - aᵀ⋅v - bᵀ⋅h
    Where v is input vector ; W is weight matrix ; h is latent vector ; a is v bias vector and b is h bias vector

    The joint probability is the exponential of -E(v,h) divided by the partition function (to transform energies in
    probabilities)

    Partition functions are computationally expensive to calculate (lots of sums) and derive (in case of
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

    def __init__(self, hidden_units: int, training=True, k=1, lr=0.01):
        """
        Args:
            hidden_units (int): Number of hidden units (latent variables)
            k (int): Number of Gibbs Samplings
            lr (float): Learning rate
        """
        super(RBMBernoulli, self).__init__()

        self.hidden_units = hidden_units
        self.k = k
        self.lr = lr

        self.training = training

    def build(self, input_shape):
        """Receive the shape of the input

                Because we're passing an image, I start flattening their shape

        Args:
            input_shape (tuple[int]): Input shape
        """

        # input_shape = NHWC = (Batch, Height, Weight, Channels)
        self.flat_shape = input_shape[1] * input_shape[2] * input_shape[3]

        # Save the original shape to reshape input
        self.width, self.height, self.channels = (
            input_shape[2],
            input_shape[1],
            input_shape[3],
        )

        self.b = tf.Variable(
            tf.zeros(shape=(self.hidden_units,)), name="b", trainable=False
        )
        self.a = tf.Variable(
            tf.zeros(shape=(self.flat_shape,)), name="a", trainable=False
        )

        # Sampling N(μ=0, σ=0.1) to initialize weights
        self.W = tf.Variable(
            tf.random.normal(
                shape=(self.flat_shape, self.hidden_units), stddev=0.3
            ),
            name="W",
            trainable=False,
        )

    def call(self, inputs, training=True):
        """Receive input and transform it

                This is the place where we call other functions to calculate conditionals probs. and reconstruct input
        Args:
            inputs (tf.Tensor): Input Tensor
        """

        inputs = tf.reshape(inputs, [-1, self.flat_shape])

        # Return the input for next RBM
        if self.training:
            return self.contrastive_divergence(inputs)

        return self.h_given_v(inputs)

    def contrastive_divergence(self, v):
        """Function to approximate the gradient where we have a positive(ϕ⁺) and negative(ϕ⁻) grad.

        ϕ⁻ = p(h₍ₜ₎ = 1 | v₍ₜ₎) ⋅ v₍ₜ₎
        ϕ⁺ = p(h₍₀₎ = 1 | v₍₀₎ ⋅ v₍₀₎

        ϕ⁺ - ϕ⁻ is the constrastive divergence which approximate the derivation of maximum log-likelihood
        """

        # Save initial input (tf.identity == np.copy)
        v_init = tf.identity(v)

        ## Gibbs Sampling
        for _ in range(self.k):
            # h ~ p(h | v)
            h = self.sample_binary_prob(self.h_given_v(v))

            # v ~ p(v | h)
            # self.v.assign(self.v_given_h(h))
            v = self.v_given_h(h)

        # p(h₍ₜ₎ = 1 | v₍ₜ₎)
        h = self.h_given_v(v)

        # p(h₍₀₎ = 1 | v₍₀₎)
        h_init = self.h_given_v(v_init)

        ## Constrastive Divergence

        # Tensordot allow to do matmul fixed in axes, in this case, fixed in batch axis
        mul_init_curr = tf.tensordot(
            v_init, h_init, axes=[0, 0]
        ) - tf.tensordot(v, h, axes=[0, 0])
        self.W.assign_add(self.lr * mul_init_curr)

        # Average each value along batch axis
        self.a.assign_add(self.lr * tf.reduce_mean(v_init - v, axis=0))
        self.b.assign_add(self.lr * tf.reduce_mean(h_init - h, axis=0))

        return h

    def v_given_h(self, h):
        """Function that implements the conditional probability:

            P(v | h) = σ( b + ∑ v⋅W )

        Args:
            Tensor: Tensor of shape [tf.shape(h)[0]]

        Returns:
            Tensor: Tensor of shape [tf.shape(v)[0]]
        """
        W_t = tf.transpose(self.W, [1, 0])
        return tf.math.sigmoid(self.a + tf.matmul(h, W_t))

    def h_given_v(self, v):
        """Function that implements the conditional probability:


            P(h | v) = σ( a + ∑ h⋅W )

        Args:
            Tensor: Tensor of shape [tf.shape(v)[0]]

        Returns:
            Tensor: Tensor of shape [tf.shape(h)[0]]
        """
        return tf.math.sigmoid(self.b + tf.matmul(v, self.W))

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

        ## We could use: return tf.reshape(tf.map_fn(lambda x: 1. if x > 0.5 else 0., probs), [-1,1])
        ## If sigmoid prob. is less then 0.5 is 0 then is 1, but it's more efficient the function below
        return tf.nn.relu(tf.sign(probs - tf.random.uniform(tf.shape(probs))))

    def get_reconstruction(self, x):
        """Function to return the reconstruction of x based on W, a and b learned previously

        Args:
            x (Tensor): Input Tensor with shape = (batch, height, width, channels)

        Returns:
            Tensor: Reconstructed Input Tensor
        """

        # Validate if image doesn't has batch dim
        try:
            if tf.rank(x) != 4:
                raise NotImplemented
        except NotImplemented as e:
            print("[!] Shape should be 4-rank!")
            sys.exit(-1)

        # Update reconstruction with new weights (Gibbs Sampling)
        h = self.h_given_v(tf.reshape(x, [-1, self.flat_shape]))
        v = self.v_given_h(h)

        # Return reconstructed input reshape to the original shape
        return tf.reshape(v, [-1, self.height, self.width, self.channels])
