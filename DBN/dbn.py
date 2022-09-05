from crbm import RBMConv
from rbm import RBMBernoulli
from utils import show_batch_images
from tqdm import tqdm
import tensorflow as tf


class DBN(tf.keras.layers.Layer):
    """Class that represents Deep Belief Network

    This network samples from joint probabilities over all RBMs.
    We can do that sampling from P(h | v) of each RBM.
    The input of each RBM is the h from the previous RBM.

    RBMs are trained separately.
    """

    def __init__(self, rbms: list, learning_decay=.1):
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

        self.learning_decay = learning_decay

    def call(self, inputs, fit=False):
        """Function to train DBN

        Args:
            inputs (Tensor): Input Tensor
            fit (bool, optional): Determine if it is suposed to train or infer each RBM

        Returns:
            Tensor: Latent/Hidden layer Tensor
        """
        for rbm in self.rbms:
            rbm.training = fit
            inputs = rbm(inputs)

        return inputs

    def get_reconstruction(self, x):
        """Function to return the reconstruction of x based on W, a and b learned previously

        Args:
            x (Tensor): Input Tensor

        Returns:
            Tensor: Reconstructed Input Tensor
        """

        # Feed forward with Gibbs Sampling
        for rbm in self.rbms:
            rbm.training = False
            x = rbm(x)

        # Here x will be h, because the reconstruction of all RBMs (except first) is the h of the previous RBM
        for rbm in reversed(self.rbms):
            x = rbm.v_given_h(x)

        return tf.clip_by_value(x, clip_value_min=0., clip_value_max=1.)

    def fit(self, inputs, epochs=1, verbose=False):
        """Function to fit and freeze the model

        Args:
            inputs (Tensor): Input Tensor
            epochs (int, optional): Number of epochs to be trained
            verbose (bool, optional): Determine if messages are shown or not (for now it's useless)
        """

        # Save original inputs
        orig_input = inputs

        for epoch in range(epochs):
            print(f"#### Epoch {epoch+1} ####")

            inputs = orig_input

            for rbm in self.rbms:
                rbm.training = True

                for i, batch in tqdm(enumerate(inputs)):
                    rbm(batch)

                # After training, freeze the model
                rbm.training = False

                # Update learning rate for next epoch
                rbm.lr *= self.learning_decay

                # Get inputs for the next RBM
                inputs = inputs.map(lambda x: rbm(
                    x)).cache().prefetch(tf.data.AUTOTUNE)
