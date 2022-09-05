import tensorflow as tf
import tensorflow_datasets as tfds
from rbm import RBMBernoulli
from crbm import RBMConv
from dbn import DBN
from utils import show_batch_images, get_datasets

# Get datatset
img_train, _, img_test, ds_info = get_datasets("cifar10")

# Input size of original data
in_size = ds_info.features["image"].shape

# Kernel size for each RBM
k_size = 3

# Get valid sizes for hidden layer of each RBM (based on kernel size)
get_hidden_size = lambda input_size, k_size: input_size - k_size + 1
first_hidden_size = get_hidden_size(in_size[0], k_size)

dbn = DBN(
    [
        RBMConv(in_size[0] // 2, 100, rbm_type="gauss", lr=1e-4),
        RBMConv(in_size[0] // 2 // 2, 25),
    ]
)

dbn.fit(img_train, epochs=2)

# Show images and their reconstructions
batch = next(iter(img_test))
x = dbn.get_reconstruction(batch)
show_batch_images(batch, x)
