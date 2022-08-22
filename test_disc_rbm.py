import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from discrete_rbm import RBMBernoulli

##❗Just for testing

# Get one image from mnist
(img,), ds_info = tfds.load(
    "mnist",
    split=["train[:10%]"],
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

original = unormalizer(batch[0]).numpy()
plt.imshow(original, cmap="gray", interpolation="nearest")
plt.show()

x = rbm.get_output(batch[0])

new = unormalizer(x).numpy()
plt.imshow(new, cmap="gray", interpolation="nearest")
plt.show()
