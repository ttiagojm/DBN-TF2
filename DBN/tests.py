import tensorflow as tf
import tensorflow_datasets as tfds
from rbm import RBMBernoulli
from crbm import RBMConv
from utils import show_batch_images


# Get images from mnist, dividing 10% for train and 5% for test
(img_train, img_test), ds_info = tfds.load(
    "mnist",
    split=["train[:10%]", "train[10%:15%]"],
    as_supervised=True,
    with_info=True,
)

in_size = ds_info.features["image"].shape


# Uncomment the variable to be used (keep only one uncommented)
#rbm_layer = [RBMBernoulli, {"hidden_units": (in_size[0] * in_size[1]) // 4, "lr": 0.1}]
rbm_layer = [RBMConv, {"hidden_units": in_size[0] - 6, "n_filters": 64}]

print("#### Using RBM: " + rbm_layer[0].__name__ + " ####\n")

# Transform ℤ: [0,255] -> ℝ: [0,1]
normalizer = tf.keras.layers.Rescaling(1.0 / 255)

# Transform ℝ: [0,1] -> ℤ: [0,255]
unormalizer = lambda x: tf.cast(tf.keras.layers.Rescaling(255)(x), tf.int32)


# Ignore y (image class), we only need the image (for now)
# If image is RGB it will be converted to grayscale
norm_img = (
    img_train.map(lambda x, y: normalizer(x))
    .cache()
    .shuffle(tf.data.experimental.cardinality(img_train))
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

# Same process as above
test_norm_img = (
    img_test.map(lambda x, y: normalizer(x))
    .cache()
    .shuffle(tf.data.experimental.cardinality(img_test))
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)


# Create a simple network with only a 1 hidden layer RBM
model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=in_size),
        # Using only a 1/4 of input units to reproduce input
        rbm_layer[0](**rbm_layer[1]),
    ]
)

# Train the model
for i, batch in enumerate(norm_img):
    print(f"{i+1} batch")
    model(batch)


# Show images and their reconstructions
batch = next(iter(test_norm_img))

rbm = model.layers[0]

# Show the first 5 images
show_batch_images(batch, unormalizer)

# Took only the first 5 images
x = rbm.get_output(batch)

# Show the first 5 reconsctructed images
show_batch_images(x, unormalizer)
