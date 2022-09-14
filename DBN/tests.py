import tensorflow as tf
import tensorflow_datasets as tfds
from rbm import RBMBernoulli
from crbm import RBMConv
from dbn import DBN
from utils import show_batch_images, get_datasets, preprocess

# Get datatset
img_train, _, img_test, ds_info = get_datasets("cifar10")
img_train = preprocess(img_train, 32, True, True)

# Input size of original data
in_size = ds_info.features["image"].shape

# Kernel size for each RBM
k_size = 5

# Get valid sizes for hidden layer of each RBM (based on kernel size)
get_hidden_size = lambda input_size, k_size: input_size - k_size + 1
first_hidden_size = get_hidden_size(in_size[0], k_size)

dbn = DBN(in_size, 5, 25, 1)

dbn.fit(img_train)

# Show images and their reconstructions
img_test = preprocess(img_test, 32, True, True)
x = dbn.reconstruct(img_test)

# Get only the images
batch = img_test.map(lambda x, y: x).cache()
x = x.map(lambda x, y: x).cache()

show_batch_images(next(iter(batch)), next(iter(x)))
