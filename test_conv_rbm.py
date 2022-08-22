import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from convolutional_rbm import RBMConv

##❗Just for testing

# Get images from mnist, dividing 10% for train and 5% for test
(img_train, img_test), ds_info = tfds.load(
    "mnist",
    split=["train[:10%]", "train[10%:15%]"],
    as_supervised=True,
    with_info=True,
)

in_size = ds_info.features["image"].shape

# Transform ℤ: [0,255] -> ℝ: [0,1]
normalizer = tf.keras.layers.Rescaling(1.0 / 255)

# Transform ℝ: [0,1] -> ℤ: [0,255]
unormalizer = lambda x: tf.cast(tf.keras.layers.Rescaling(255)(x), tf.int32)

# Ignore y (image class), we only need the image (for now)
# If image is RGB it will be converted to grayscale
norm_img = (
    img_train.map(
        lambda x, y: normalizer(tf.image.rgb_to_grayscale(x))
        if in_size[-1] == 3
        else normalizer(x)
    )
    .cache()
    .shuffle(tf.data.experimental.cardinality(img_train))
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

# Same process as above
test_norm_img = (
    img_test.map(
        lambda x, y: normalizer(tf.image.rgb_to_grayscale(x))
        if in_size[-1] == 3
        else normalizer(x)
    )
    .cache()
    .shuffle(tf.data.experimental.cardinality(img_test))
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)


# New input size (in case some RGB to Grayscale happened)
in_size = tf.shape(next(iter(norm_img))[0])


# Create a simple network with only a 1 hidden layer RBM
model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=in_size),
        RBMConv(hidden_units=in_size[0] - 10, n_filters=32),
    ]
)

# Train the model
for i, batch in enumerate(norm_img):
    print(f"{i+1} batch")
    model(batch)


# Show images and its reconstruction

for i, batch in enumerate(test_norm_img):

    # Show only 5 batches, 1 image per batch
    if i == 5:
        break

    # Get RBM layer
    rbm = model.layers[0]

    original = unormalizer(batch[0]).numpy()
    plt.imshow(original, cmap="gray", interpolation="nearest")
    plt.show()

    # Tensor comes with 4D dimension (first is Batch dim which we remove)
    x = rbm.get_output(batch[0])[0]

    new = unormalizer(x).numpy()
    plt.imshow(new, cmap="gray", interpolation="nearest")
    plt.show()
