import tensorflow as tf
import tensorflow_datasets as tfds
from rbm import RBMBernoulli
from crbm import RBMConv
from dbn import DBN
from utils import show_batch_images


(img_train, img_test), ds_info = tfds.load(
    "mnist",
    split=["train[:10%]", "train[10%:15%]"],
    as_supervised=True,
    with_info=True,
)

in_size = ds_info.features["image"].shape

# Be careful with this value.
# It will be used sequentially in each RBM, e.g knowing that in_size[0] = 28 and dec_val = 6:
# RBM 1 -> hidden_units = in_size[0] - dec_val = 22
# RBM 2 -> RBM_1_hidden_units - dec_val = 16
# We can conclude that dec_val < in_size[0] // 2, in case that we have only 2 RBMs
dec_val = in_size[0] // 2 - 1

# Uncomment the variable to be used (keep only one uncommented)
#rbm_layer = [RBMBernoulli, {"hidden_units": (in_size[0] * in_size[1]) // 4, "lr": 0.1}]
rbm_layer = [RBMConv, {"hidden_units": in_size[0] - dec_val, "n_filters": 32}]
dbn_layer = [DBN, {"rbm_type": rbm_layer[0].__name__, "rbm_num":2, "decrease_val": dec_val, "rbm_params":rbm_layer[1]}]

print("#### Using RBM: " + rbm_layer[0].__name__ + " ####\n")

# Transform ℤ: [0,255] -> ℝ: [0,1]
normalizer = tf.keras.layers.Rescaling(1.0 / 255)

# Transform ℝ: [0,1] -> ℤ: [0,255]
unormalizer = lambda x: tf.cast(tf.keras.layers.Rescaling(255)(x), tf.int32)


# Ignore y (image class), we only need the image (for now)
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
        #rbm_layer[0](**rbm_layer[1]),
        dbn_layer[0](**dbn_layer[1])
    ]
)

# Train the model
for epoch in range(1):
    print(f"#### Epoch {epoch+1} ####")
    for i, batch in enumerate(norm_img):
        print(f"{i+1} batch")
        model(batch)


# Show images and their reconstructions
batch = next(iter(test_norm_img))

rbm = model.layers[0]
x = rbm.get_reconstruction(batch)

# Show the first 5 images
show_batch_images(batch, x, unormalizer)
