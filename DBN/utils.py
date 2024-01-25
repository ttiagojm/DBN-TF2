import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
from exceptions import MismatchShape
from datetime import datetime


def get_dataset(batch_size):
    """Download and normalize dataset, transforming values in Z: [0, 255] to R: [0,1] 

    Args:
        batch_size (int): Batch size
    
    Returns:
        tf.keras.Dataset: Training Dataset
        tf.keras.Dataset: Test Dataset
        tuple: Images shape
    """

    (ds_train, ds_test), ds_info = tfds.load(
        'cifar10',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True
    )

    def normalize_img(image, label):
        return tf.cast(image, tf.float32) / 255.0, label

    ds_train = ds_train.map(normalize_img)
    ds_test = ds_test.map(normalize_img)

    # Drop remainder because CRBM needs to receive exactly batch_size batches
    ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples).cache().batch(batch_size, drop_remainder=True)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
    
    ds_test = ds_test.cache().batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_test, ds_info.features["image"].shape


class TimingCallback(tf.keras.callbacks.Callback):
    """
        Callback for .fit() method of a tf.keras.Model class

        This callback will count the time elapsed for each epoch and save in history object
    """
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = datetime.now()

    def on_epoch_end(self, epoch, logs=None):
        epoch_end_time = datetime.now()
        logs['epoch_times'] = (epoch_end_time - self.epoch_start_time).total_seconds()


def test_net(ds, ds_test, input_shape):
    """Create a CNN, train and test with the passed datasets

    Args:
        ds (tf.keras.Dataset): Train Dataset
        ds_test (tf.keras.Dataset): Teste Dataset
        input_shape (tuple): Input shape
    
    Returns:
        tf.keras.callbacks.History: History object returned from .fit() method
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    timing_callback = TimingCallback()

    # Return history data
    return model.fit(ds, epochs=10, validation_data=ds_test, callbacks=[timing_callback])


def transform(cdbn, x):
    """Transform input using a CDBN

    Args:
        cdbn (CDBN): Object of CDBN
        x (tf.keras.Dataset): Batched Dataset
    
    Returns:
        tf.keras.Dataset: Transformed input
    """
    x = cdbn.encode(x)
    return cdbn.decode(x)


def preprocessing(cdbn, ds_train, ds_test):
    """Apply transformation in each dataset, using CDBN 

    Args:
        cdbn (CDBN): Object of CDBN
        ds_train (tf.keras.Dataset): Train Dataset
        ds_test (tf.keras.Dataset): Train Dataset
    
    Returns:
        tf.keras.Dataset: Transformed Train Dataset
        tf.keras.Dataset: Transformed Train Dataset
    """
    transformed_train_data = [(transform(cdbn, x), y) for x, y in ds_train]
    transformed_test_data = [(transform(cdbn, x), y) for x, y in ds_test]

    train_images, train_labels = zip(*transformed_train_data)
    test_images, test_labels = zip(*transformed_test_data)

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    # Transform each array in a Dataset again
    ds_transformed_train = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).cache().prefetch(tf.data.experimental.AUTOTUNE)
    ds_transformed_test = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).cache().prefetch(tf.data.experimental.AUTOTUNE)

    return ds_transformed_train, ds_transformed_test


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