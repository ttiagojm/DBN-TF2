import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import sys
import os
from time import time
from inspect import signature, _empty


# Transform ℤ: [0,255] -> ℝ: [0,1]
normalizer = tf.keras.layers.Rescaling(1.0 / 255)

# Transform ℝ: [0,1] -> ℤ: [0,255]
unormalizer = lambda x: tf.cast(tf.keras.layers.Rescaling(255)(x), tf.int32)


def preprocess(
    ds: tf.data.Dataset,
    batch_size: int,
    labels: bool,
    normalize: bool,
    autoencoder=None,
):
    """Function to preprocess data

       Its needed to be able to map the 3 datasets.

    Args:
        ds (tf.data.Dataset): Dataset to iterate
        batch_size (int): Number of images per batch
        labels (bool): Boolean to determine if labels should be returned
        normalize (bool, optional): Determine if the dataset should or shouldn't be normalized
        autoencoder (function, optional): Autoencoder function/layer

    Returns:
        tuple: Tuple of processed train, validation and test data tensors
    """
    # Normalize Dataset or just return it
    norm_or_not = lambda x: normalizer(x) if normalize else x

    # The order changes if autoencoder is passed because autoencoder needs that the Dataset is batched
    if autoencoder is None:
        result = (
            ds.map(
                lambda x, y: (norm_or_not(x), y) if labels else norm_or_not(x)
            )
            .cache()
            .shuffle(tf.data.experimental.cardinality(ds))
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
    else:
        result = (
            ds.cache()
            .shuffle(tf.data.experimental.cardinality(ds))
            .batch(batch_size)
            .map(
                lambda x, y: (autoencoder(norm_or_not(x)), y)
                if labels
                else autoencoder(norm_or_not(x))
            )
            .prefetch(tf.data.AUTOTUNE)
        )

    return result


def get_datasets(
    dataset_name: str,
    batch_size: int,
    normalize=True,
    labels=False,
    autoencoder=None,
):
    """Function get a dataset and prepare it to be used to efficiently on training pipeline

    Args:
        dataset_name (str): Name of the dataset of tensorflow_datasets
        batch_size (int): Number of images per batch
        normalize (bool, optional): Determine if the dataset should or shouldn't be normalized
        labels (bool, optional): Boolean to determine if labels should be returned
        autoencoder (function, optional): Autoencoder function/layer

    Return:
        tuple: Tuple of processed train, validation and test data tensors
    """
    (img_train, img_val, img_test), ds_info = tfds.load(
        dataset_name,
        split=["train[:80%]", "train[80%:]", "test"],
        as_supervised=True,
        with_info=True,
    )

    norm_train, norm_val, norm_test = map(
        lambda x: preprocess(x, batch_size, labels, normalize, autoencoder),
        [img_train, img_val, img_test],
    )
    return norm_train, norm_val, norm_test, ds_info


"""
	Helper functions/classes for RBMs 

"""


def show_batch_images(batch, pred, num_imgs=5, unormalize=True):
    """Function to show original and reconstructed images side by side
            given a batch of images.

    Args:
        batch (Tensor): Tensor with a batch of original images
        pred (Tensor): Tensor with a batch of reconstructed images
        num_imgs (int, optional): Number of images to show from batch
        unormalize (bool, optional): Unnormalize images
    """
    for img, img_pred in zip(batch[:num_imgs], pred[:num_imgs]):

        if unormalize:
            i, i_pred = unormalizer(img).numpy(), unormalizer(img_pred).numpy()
        else:
            i, i_pred = img.numpy(), img_pred.numpy()

        fig, (ax0, ax1) = plt.subplots(1, 2)
        ax0.imshow(i, interpolation="nearest")
        ax1.imshow(i_pred, interpolation="nearest")
        plt.show()


"""
	Helper functions/classes for DBN
"""


def get_shallow_net(in_size: tuple[int]):
    """Function to create and compile a shallow convnet for training

    Args:
        in_size (tuple[int]): Input size

    Returns:
        Model: Prepared Sequential Model
    """
    shallow = tf.keras.Sequential(
        [
            tf.keras.Input(shape=in_size),
            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", padding="same"
            ),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(
                64, (3, 3), activation="relu", padding="same"
            ),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10),
        ]
    )

    shallow.compile(
        optimizer=tf.keras.optimizers.SGD(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return shallow


def set_tensorboard_weights():
    """Create a Summary Writter

    Returns:
        tf.summary: Writter to log histograms
    """
    base_dir = os.path.abspath(os.path.dirname(__file__))
    return tf.summary.create_file_writer(os.path.join(base_dir, "Tensorboard"))


def write_tensorboard_weights(writer, weights: tf.Tensor, name: str):
    """Write weights to plot on Tensorboard Histograms

    Args:
        writer (tf.summary): Summary writter
        weights (tf.Tensor): Tensor with weights
        name (str): Name of histogram
    """
    with writer.as_default():
        tf.summary.histogram(name, weights)
