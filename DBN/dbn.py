from crbm import RBMConv
from rbm import RBMBernoulli
from utils import show_batch_images, preprocess
from exceptions import (
    MismatchCardinality,
    NonSquareInput,
)
from tqdm import tqdm
import tensorflow as tf


class DBN(tf.keras.Model):
    """Class that represents Deep Belief Network

    This network samples from joint probabilities over all RBMs.
    We can do that sampling from P(h | v) of each RBM.
    The input of each RBM is the h from the previous RBM.

    RBMs are trained separately.
    """

    calc_hidden_units = lambda in_size, k_size: in_size - k_size + 1

    def __init__(
        self,
        in_size: tuple[int, int, int],
        k_size: int,
        n_filters: int,
        epochs=2,
    ):
        super(DBN, self).__init__()

        # Input size validator (h,w,c)
        if len(in_size) != 3:
            raise MismatchCardinality(type(in_size))

        # Square Input (h==w)
        if in_size[0] != in_size[1]:
            raise NonSquareInput()

        # Simple 2 Hidden Layer CRBM Model (1 real layer and 1 binary layer, no regularization)
        real_hidden_size = DBN.calc_hidden_units(in_size[0], k_size)
        self.real_latent = RBMConv(real_hidden_size, n_filters, sigma=1.0)

        bin_hidden_size = DBN.calc_hidden_units(real_hidden_size, k_size)
        self.bin_latent = RBMConv(bin_hidden_size, n_filters)

        self.epochs = epochs

    def fit(self, inputs):
        # Make sure that inputs have batches of size <= 10
        inputs = preprocess(
            inputs,
            10,
            labels=False,
            normalize=False,
            unbatch=True,
            shuffle=False,
        )

        # Save original inputs
        orig_input = inputs

        for epoch in range(self.epochs):
            print(f"#### Epoch {epoch+1} ####")

            # Reset inputs each epoch
            inputs = orig_input

            for rbm in [self.real_latent, self.bin_latent]:
                for i, batch in tqdm(enumerate(inputs)):
                    rbm.fit(batch)

                # Get inputs (h) for the next RBM
                inputs = (
                    inputs.map(
                        lambda x: rbm(x), num_parallel_calls=tf.data.AUTOTUNE
                    )
                    .cache()
                    .prefetch(tf.data.AUTOTUNE)
                )

    def call(self, inputs):
        # Encoder
        real_hidden = self.real_latent(inputs)
        bin_hidden = self.bin_latent(real_hidden)

        # Decoder
        bin_input = self.bin_latent.v_given_h(bin_hidden)
        real_input = self.real_latent.v_given_h(bin_input)

        return real_input

    def reconstruct(self, inputs):
        return (
            inputs.map(
                lambda x, y: (self(x), y), num_parallel_calls=tf.data.AUTOTUNE
            )
            .cache()
            .prefetch(tf.data.AUTOTUNE)
        )
