import tensorflow as tf
import tensorflow_datasets as tfds
from rbm import RBMBernoulli
from crbm import RBMConv
from dbn import DBN
from utils import show_batch_images, get_datasets

img_train, _ , img_test, ds_info = get_datasets("fashion_mnist")

in_size = ds_info.features["image"].shape


# Be careful with this value.
# It will be used sequentially in each RBM, e.g knowing that in_size[0] = 28 and dec_val = 6:
# RBM 1 -> hidden_units = in_size[0] - dec_val = 22
# RBM 2 -> RBM_1_hidden_units - dec_val = 16
# We can conclude that dec_val < in_size[0] // 2, in case that we have only 2 RBMs
dec_val = 4

dbn = DBN([
            RBMConv(in_size[0] - dec_val, 8, lr=.1),
            RBMConv(in_size[0] - dec_val*2, 8, lr=.1)
        ])

dbn.fit(img_train)

# Show images and their reconstructions
batch = next(iter(img_test))

x = dbn.get_reconstruction(batch[0])

# Show the first 5 images
show_batch_images(batch[0], x)