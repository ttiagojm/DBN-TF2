import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from discrete_rbm import RBMBernoulli

##❗Just for testing

# Get one image from mnist
(img,), ds_info = tfds.load(
    "mnist",
    split=["train[:1]"],
    as_supervised=True,
    with_info=True,
)

in_size = ds_info.features["image"].shape

# Transform ℤ: [0,255] -> ℝ: [0,1]
normalizer = tf.keras.layers.Rescaling(1.0 / 255)


# Transform ℝ: [0,1] -> ℤ: [0,255]
unormalizer = lambda x: tf.cast(tf.keras.layers.Rescaling(255)(x), tf.int32)

# Ignore y (image class), we only need the image
# Batch is useless in this example, but we use it to keep stuff organized for final version
norm_img = img.map(lambda x, y: normalizer(x)).batch(32)


# Create a simple network with only a 1 hidden layer RBM
model = tf.keras.Sequential(
    [tf.keras.Input(shape=in_size), RBMBernoulli(hidden_units=6000)]
)

# Pass the image to the model
for i, batch in enumerate(norm_img):
    print(f"{i+1} batch")

    # Show original image
    original = unormalizer(batch[0]).numpy()
    plt.imshow(original, cmap="gray", interpolation='nearest')
    plt.show()

    x = model(batch)

    # Show reconstructed image (gray is inverted because prob = 1 means pixel colored and prob = 0 pixel uncolored)
    # Because 255 * 1 = 255 which is white and 255 * 0 = 0 which is black, we invert the grayscale here
    new = unormalizer(x).numpy()
    plt.imshow(new, cmap="gray_r", interpolation='nearest')
    plt.show()
