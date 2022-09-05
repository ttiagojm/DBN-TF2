import tensorflow as tf
import tensorflow_probability as tfp
import sys

# Alias to access distributions module
tfd = tfp.distributions


class RBMConv(tf.keras.layers.Layer):
    """Class that represents a Convolutional Restricted Boltzman Machine

    RBMs are represented by an Energy Function -> E(v,h) = ∑ᴷₖ₌₁ hᵏ ⋅ (Wᵏ * v) - ∑ᴷₖ₌₁ bₖ ⋅ hᵏ - a ⋅ ∑ᵢⱼ vᵢⱼ
    Where v is input vector ; W is a kernel ; h is latent vector ; a is v bias vector and b is h bias vector

    Knowing that v has shape NᵥxNᵥ and h NₕxNₕxK then W should has shape Nᵥ-Nₕ+1 x Nᵥ-Nₕ+1 x K

    Convolution has no padding (valid padding) and stride = 1 and produces K feature maps

    The joint probability is the exponential of -E(v,h) divided by the partition function (to transform energies in
    probabilities)

    Partition functions are computationally expensive to calculate (lots of sums) and derive (in case of
    deriving the max log-likelihood)

    To prevent that conditionals are used to:
            * Map Input (v) into latent space (h)
            * Reconstruct Input (v) using latent space (h)

    Gibbs Sampling does the job and Contrastive Divergence will be used to approximate gradients.

    P(h=1 | v) = σ( (Wᵏ * v) + bₖ)
    P(v=1 | h) = σ( (∑ᴷ Wᵏ * h) + a )

    The conditional above are using binary visible units (v), in case of real-valued visible units:
    P(h=1 | v) = N(((Wᵏ * v) + bₖ), sigma²), where N is a Multivariate Gaussian dist.
    P(v=1 | h) = sigmoid( ((∑ᴷ Wᵏ * h) + a)/sigma² )

    A more complete version has a Max Pooling Layer too (it's not implemented, yet)

    References: (Lee, Grosse, Ranganath, Ng, 2009)
                (Lee, Honglak and Ekanadham, Chaitanya and Ng, Andrew, 2007)
    """

    def __init__(
        self,
        hidden_units: int,
        n_filters: int,
        rbm_type="binary",
        training=True,
        k=1,
        lr=1e-3,
    ):
        """
        Args:
            hidden_units (int): Number of hidden units (latent variables)
            n_filters (int): Number of filters (latent groups)
            rbm_type(str, optional): Distinguish between binary and real valued visible units
            training(bool, optional): Boolean to know if RBM is able to train or just infer
            k (int, optional): Number of Gibbs Samplings
            lr (float, optional): Learning rate
        """
        super(RBMConv, self).__init__()

        self.h = tf.Variable(
            tf.zeros(shape=(hidden_units, hidden_units, n_filters)),
            trainable=False,
        )
        self.b = tf.Variable(tf.zeros(shape=(n_filters,)), trainable=False)

        self.n_filters = n_filters
        self.lr = lr
        self.k = k
        self.sigma = 0.1
        self.training = training
        self.v_activation = (
            tf.math.sigmoid
            if rbm_type == "binary"
            else self.sample_gaussian_prob
        )

    def build(self, input_shape):
        """Receive the shape of the input

            input_shape has a shape = (Batch, Height, Weight, Channels)

        Args:
            input_shape (tuple[int]): Input shape
        """

        # Validate if image is square
        try:
            if input_shape[1] != input_shape[2]:
                raise NotImplemented
        except NotImplemented as e:
            print(
                f"[!] Expect [{input_shape[1]},{input_shape[1]}]. Got [{input_shape[1]},{input_shape[2]}]"
            )
            sys.exit(-1)

        self.a = tf.Variable(0.01, trainable=False)

        # Sampling N(μ=0, σ=.01) to initialize weights
        # Don't Forget W shape: Nᵥ-Nₕ+1 x Nᵥ-Nₕ+1 x K
        # Based on Tensorflow Docs kernel should follow this shape:
        # [filter_height, filter_width, in_channels, out_channels]
        self.W = tf.Variable(
            tf.random.normal(
                shape=(
                    input_shape[1] - tf.shape(self.h)[1] + 1,
                    input_shape[1] - tf.shape(self.h)[1] + 1,
                    input_shape[3],
                    self.n_filters,
                ),
                stddev=0.01,
            ),
            trainable=False,
        )

    def call(self, inputs):
        """Receive input and transform it

                This is the place where we call other functions to calculate conditionals and reconstruct input
        Args:
            inputs (tf.Tensor): Input Tensor
        """

        self.batch_size = tf.shape(inputs)[0]
        self.v_shape = tf.shape(inputs)

        # Return h as input for next RBM
        if self.training:
            return self.contrastive_divergence(inputs)

        return self.sample_binary_prob(self.h_given_v(inputs))

    def contrastive_divergence(self, v):
        """Function to approximate the gradient where we have a positive(ϕ⁺) and negative(ϕ⁻) grad.

        ϕ⁻ = p(h₍ₜ₎ = 1 | v₍ₜ₎) ⋅ v₍ₜ₎
        ϕ⁺ = p(h₍₀₎ = 1 | v₍₀₎ ⋅ v₍₀₎

        ϕ⁺ - ϕ⁻ is the constrastive divergence which approximate the derivation of maximum log-likelihood
        """

        # Save original image
        v_init = tf.identity(v)

        ## Gibbs Sampling
        for _ in range(self.k):
            # h ~ p(h | v), we are sampling a binary variable from a Bernoulli dist.
            h = self.sample_binary_prob(self.h_given_v(v))

            # v ~ p(v | h)
            v = self.v_given_h(h)

        # p(h₍₀₎ = 1 | v₍₀₎, not sampling just calculating probs.
        h_init = self.h_given_v(v_init)

        # p(h₍ₜ₎ = 1 | v₍ₜ₎)
        h = self.h_given_v(v)

        # Here we use a trick to transform h in a kernel and being able to use batch convolution
        # v and h have the same batch size, so we put batch size as input channel. On v we put the
        # real input channel as batch size
        grad_w_0 = tf.nn.conv2d(
            tf.transpose(v_init, [3, 1, 2, 0]),
            tf.transpose(h_init, [1, 2, 0, 3]),
            strides=1,
            padding="VALID",
        )
        grad_w_t = tf.nn.conv2d(
            tf.transpose(v, [3, 1, 2, 0]),
            tf.transpose(h, [1, 2, 0, 3]),
            strides=1,
            padding="VALID",
        )

        # In the end we have a grad shape = [input channel, height, width, output channel]
        # Where input channel is the real one
        grad = tf.transpose(grad_w_0 - grad_w_t, [1, 2, 0, 3])

        # Needs to be divided by batch size because all convolutions where applied in batch size depth
        # Not in their real depth (input channel)
        self.W.assign_add(
            self.lr * (grad / tf.cast(self.batch_size, dtype=tf.float32))
        )

        # Because a is a single value bias for the input tensor all init values where summed then subtracted
        # with the sum of current input values
        self.a.assign_add(
            self.lr * (tf.reduce_mean(v_init - v, axis=[0, 1, 2]))[0]
        )

        # Same logic as a but here b has n_filter lines so we sum along all axes except the channel one
        self.b.assign_add(
            self.lr * (tf.reduce_mean(h_init - h, axis=[0, 1, 2]))
        )

        return self.sample_binary_prob(self.h_given_v(v_init))

    def v_given_h(self, h):
        """Function that implements the conditional probability:

            P(v | h) = σ( (W * h) + a )

            Because most of times h is smaller than W and because we are mapping the new probability model, H,
            to our Input (reconstructing it). So we need a deconvolution (transposed convolution) because to create h
            value we convolved v and W, so now we deconvolved h and W to reconstruct v.

            Transpose Convolution is the inverse of Convolution, while convs. downsample an images, transposed convs.
            upscale them.

            Strides and Padding must be the same as in the Convolution. If Output shape is wrongly calculated by us,
            Tensorflow will raise an error.

            v_activation is our σ function and it's needed since CRBM can be real or binary valued. In case of real-valued, sigmoid
            is replaced by a Gaussian distribution with mean equal to the result of the convolution and the variance
            will be sigma²

        Returns:
            Tensor: Tensor of shape [batch_size, Nv, Nv, channels]
        """
        return self.v_activation(
            (
                self.a
                + tf.nn.conv2d_transpose(
                    h,
                    self.W,
                    output_shape=self.v_shape,
                    strides=1,
                    padding="VALID",
                )
            )
        )

    def h_given_v(self, v):
        """Function that implements the conditional probability:

            P(h | v) = σ( (Wᵏ * v) + bₖ)


            W should be reversed in x and y axis because in v_given_h we use it normally and this is the oposite
            operation.

            If our CRBM is real-valued, it's needed to divide the result by sigma²

        Args:
            Tensor: Tensor of shape [batch_size, Nw, Nw, n_filters]

        Returns:
            Tensor: Desc
        """
        return tf.math.sigmoid(
            (
                self.b
                + tf.nn.conv2d(
                    v,
                    tf.reverse(self.W, axis=[0, 1]),
                    strides=1,
                    padding="VALID",
                )
            )
            / (
                tf.square(self.sigma)
                if self.v_activation == self.sample_gaussian_prob
                else 1.0
            )
        )

    def sample_gaussian_prob(self, mean):
        """Function to create and sample from a Multivariate Gaussian distribution

            Mean (loc) is the relation bewteen H and V given by the deconvolution
            Standard Deviation (scale_diag) is the sigma²

        Args:
            mean (Tensor): Result of a deconvolution and used as mean to the distribution

        Returns:
            Tensor: New Tensor with all new samples
        """
        dist = tfd.MultivariateNormalDiag(
            loc=mean, scale_diag=tf.fill(tf.shape(mean), tf.square(self.sigma))
        )

        gauss = dist.sample()

        # print(tf.reduce_min(gauss), tf.reduce_max(gauss))

        return gauss

    def sample_binary_prob(self, probs):
        """Function that samples from a Bernoulli distribution

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
            x (Tensor): Input Tensor

        Returns:
            Tensor: Reconstructed Input Tensor
        """

        # Validate if image doesn't has batch dim
        try:
            if len(tf.shape(x)) != 4:
                raise NotImplemented
        except NotImplemented as e:
            print(
                f"[!] Wrong shape! Should be [Batch, Height, Width, Channels]. Got [{tf.shape(x)}]"
            )
            sys.exit(-1)

        # Needs to be updated for some calculations
        self.v_shape = tf.shape(x)

        h = self.sample_binary_prob(self.h_given_v(x))
        v = self.v_given_h(h)

        # Return reconstructed input reshape to the original shape
        return v
