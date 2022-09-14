import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import sys
import os
from inspect import signature, _empty
from exceptions import MismatchShape


# Transform ℤ: [0,255] -> ℝ: [0,1]
normalizer = tf.keras.layers.Rescaling(1.0 / 255)

# Transform ℝ: [0,1] -> ℤ: [0,255]
unormalizer = lambda x: tf.cast(tf.keras.layers.Rescaling(255)(x), tf.int32)


def preprocess(
    ds: tf.data.Dataset,
    batch_size: int,
    labels: bool,
    normalize: bool,
    unbatch=False,
    shuffle=True,
) -> tf.data.Dataset:
    """Function to preprocess data

       Its needed to be able to map the 3 datasets.

    Args:
        ds (tf.data.Dataset): Dataset to iterate
        batch_size (int): Number of images per batch
        labels (bool): Boolean to determine if labels should be returned
        normalize (bool): Determine if the dataset should or shouldn't be normalized
        unbatch (bool, optional): Unbatch the dataset if True
        shuffle (bool, optional): Shuffle the dataset if True

    """
    # Normalize Dataset or just return it
    norm_or_not = lambda x: normalizer(x) if normalize else x

    if unbatch:
        ds = ds.unbatch()

    if shuffle:
        ds = ds.shuffle(tf.data.experimental.cardinality(ds))

    return (
        ds.batch(batch_size)
        .map(
            lambda x, y: (norm_or_not(x), y) if labels else norm_or_not(x),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )


def get_datasets(dataset_name: str):
    """Function get a dataset

    Args:
        dataset_name (str): Name of the dataset of tensorflow_datasets

    Return:
        tuple: Tuple of train, validation and test data tensors
    """
    (img_train, img_val, img_test), ds_info = tfds.load(
        dataset_name,
        split=["train[:80%]", "train[80%:]", "test"],
        as_supervised=True,
        with_info=True,
    )

    return img_train, img_val, img_test, ds_info


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



def check_shape(shape_1, shape_2):
    """Check if 2 TensorShapes are equal
    
    Args:
        shape_1 (TensorShape): First shape
        shape_2 (TensorShape): Second shape
    
    Raises:
        MismatchShape: If they're different an Exception is raised
    """
    if not tf.reduce_all(tf.equal(shape_1, shape_2)):
        raise MismatchShape(shape_1, shape_2)