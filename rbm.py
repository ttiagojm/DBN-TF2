import tensorflow as tf
import tensorflow_datasets as tfds


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
        self.h = tf.Variable( tf.zeros(shape=(hidden_units, 1)), name='h')
        self.b = tf.Variable( tf.zeros(shape=(hidden_units, 1)), name='b')

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
        self.a = tf.Variable( tf.zeros(shape=(self.flat_shape, 1)), name='a')
        self.W = tf.Variable( tf.keras.initializers.GlorotNormal()(
            shape=(self.flat_shape, tf.shape(self.h)[0])
        ), name='W')

    def call(self, inputs):
        """Receive input and transform it

                This is the place where we call other functions to calculate conditionals and reconstruct input
                
                NOTE: Batch are not supported for now
        Args:
            inputs (tf.Tensor): Input Tensor
        """

        # 📝TODO: assure that inputs shape are always equal to flat_shape
        for b in range(tf.shape(inputs)[0]):
            self.v = tf.reshape(inputs[b], [self.flat_shape, 1])
            self.k_gibbs_sampling()
            self.contrastive_divergence()

        # Return reconstructed input
        return self.v

    def contrastive_divergence(self):
        h_init = self.h_given_v(self.v_init)

        self.W.assign(self.lr * (
            tf.linalg.matmul(self.v_init, tf.transpose(h_init))
            - tf.linalg.matmul(self.v, tf.transpose(self.h))
        ))
        self.a.assign(self.lr * (self.v_init - self.v))
        self.b.assign(self.lr * (h_init - self.h))

    def k_gibbs_sampling(self):
        # Save initial input (tf.identity == np.copy)
        self.v_init = tf.identity(self.v)

        for _ in range(self.k):
            self.h = self.h_given_v(self.v)
            self.v = self.v_given_h()

    def v_given_h(self):
        return tf.math.sigmoid(self.a + tf.linalg.matmul(self.W, self.h))

    def h_given_v(self, v):
        return tf.math.sigmoid(
            self.b + tf.linalg.matmul(tf.transpose(self.W), v)
        )


##❗Just for testing

# Get one image from cifar-10
(img,), ds_info = tfds.load(
    "cifar10",
    split=["train[:1]"],
    as_supervised=True,
    with_info=True,
)

in_size = ds_info.features["image"].shape

normalizer = tf.keras.layers.Rescaling(1.0 / 255)
unormalizer = lambda x: tf.cast( tf.keras.layers.Rescaling(255)(x), tf.int32 )

# Ignore y (image class), only need the image
# Batch is useless in this example, but we use it to keep stuff organized for final version
norm_img = img.map(lambda x, y: normalizer(x)).batch(32)

model = tf.keras.Sequential(
    [tf.keras.Input(shape=in_size), RBMBernoulli(hidden_units=28)]
)

# Pass each image to the model
for batch in norm_img:
    model(batch)
