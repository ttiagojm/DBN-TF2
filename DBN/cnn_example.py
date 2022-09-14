import tensorflow as tf
import tensorflow_datasets as tfds
from utils import get_datasets, preprocess
from dbn import DBN
from crbm import RBMConv

"""
    An example to compare a shallow convnet trained with original and autoencoded images 
"""

# Get dataset original images
dataset = "cifar10"
img_train, img_val, _, ds_info = get_datasets(dataset)

# Preprocess datasets
img_train, img_val = map(
    lambda x: preprocess(x, 32, True, True), [img_train, img_val]
)

# Input original size
in_size = ds_info.features["image"].shape

# Kernel and filter size for RBMs
k_size = 5
n_filters = 25


# Create autoencoder
dbn = DBN(in_size, k_size, n_filters, 2)

# Train autoencoder
dbn.fit(img_train)

# Get new reconstructed images
recon_train = dbn.reconstruct(img_train)
recon_val = dbn.reconstruct(img_val)

# Train a model with new images
shallow = tf.keras.Sequential(
    [
        tf.keras.Input(shape=in_size),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
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

shallow.fit(recon_train, validation_data=recon_val, epochs=10)


# Train a model with original images
shallow = tf.keras.Sequential(
    [
        tf.keras.Input(shape=in_size),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
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

shallow.fit(img_train, validation_data=img_val, epochs=10)
