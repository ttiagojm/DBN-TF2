import tensorflow as tf
import tensorflow_datasets as tfds
from rbm import RBMBernoulli
from utils import show_batch_images

##❗Just for testing Discrete RBM

# Get images from mnist
(img,), ds_info = tfds.load(
    "mnist",
    split=["train[:5%]"],
    as_supervised=True,
    with_info=True,
)

in_size = ds_info.features["image"].shape

# Transform ℤ: [0,255] -> ℝ: [0,1]
normalizer = tf.keras.layers.Rescaling(1.0 / 255)

# Transform ℝ: [0,1] -> ℤ: [0,255]
unormalizer = lambda x: tf.cast(tf.keras.layers.Rescaling(255)(x), tf.int32)

# Ignore y (image class), we only need the image (for now)
norm_img = (
    img.map(lambda x, y: normalizer(x))
    .cache()
    .shuffle(tf.data.experimental.cardinality(img))
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)


# Create a simple network with only a 1 hidden layer RBM
model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=in_size),
        # Using only a 1/4 of input units to reproduce input
        RBMBernoulli(hidden_units=(in_size[0] * in_size[1]) // 4, lr=0.1),
    ]
)

# Train the model
for i, batch in enumerate(norm_img):
    print(f"{i+1} batch")
    model(batch)


# Show image and its reconstruction
# [!] Warning: We shouldn't use train images, but for this kind of example is irrelevant
batch = next(iter(norm_img))

rbm = model.layers[0]

# Show the first 5 images
show_batch_images(batch, unormalizer)

# Took only the first 5 images
x = rbm.get_output(batch)

# Show the first 5 reconsctructed images
show_batch_images(x, unormalizer)
