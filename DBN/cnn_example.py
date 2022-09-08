import tensorflow as tf
import tensorflow_datasets as tfds
from utils import get_datasets, get_shallow_net
from dbn import DBN
from crbm import RBMConv

"""
	An example to compare a shallow convnet trained with original and autoencoded images 
"""

# Get dataset original images
dataset = "cifar10"
img_train, img_val, _, ds_info = get_datasets(dataset, 32, labels=True)

# Epochs for shallow nets to train
epochs = 2

# Input original size
in_size = ds_info.features["image"].shape

# Kernel and filter size for RBMs
k_size = 5
n_filters = 25

print("#### Model trained with original images ####\n")

# Create a shallow ConvNet using original images
shallow = get_shallow_net(in_size)

hist_orig = shallow.fit(img_train, validation_data=img_val, epochs=epochs)

print(
    "Acc: %.3f"
    % (sum(hist_orig.history["accuracy"]) / len(hist_orig.history["accuracy"]))
)
print(
    "Val_Acc: %.3f"
    % (
        sum(hist_orig.history["val_accuracy"])
        / len(hist_orig.history["val_accuracy"])
    )
)
print(
    "Loss: %.3f"
    % (sum(hist_orig.history["loss"]) / len(hist_orig.history["loss"]))
)
print(
    "Val_Loss: %.3f"
    % (sum(hist_orig.history["val_loss"]) / len(hist_orig.history["val_loss"]))
)


print("#### Model trained with reconstructed images ####\n")

# Get valid sizes for hidden layer of each RBM (based on kernel size)
get_hidden_size = lambda input_size, k_size: input_size - k_size + 1
first_hidden_size = get_hidden_size(in_size[0], k_size)
sec_hidden_size = get_hidden_size(first_hidden_size, k_size)

pre_train = tf.keras.Sequential(
    [
        DBN(
            [
                RBMConv(first_hidden_size, n_filters, sigma=1.0),
                RBMConv(sec_hidden_size, n_filters),
            ]
        )
    ]
)

# Train the DBN model (keep batch as 10, bigger batches will mess with autoencoder estimations)
img_train = img_train.map(lambda x, y: x).unbatch().cache().batch(10).prefetch(tf.data.AUTOTUNE)
pre_train.layers[0].fit(img_train, epochs=2)

# Recreate dataset with reconstructed images
img_train, img_val, _, ds_info = get_datasets(
    dataset, 32, labels=True, autoencoder=pre_train
)

## Create a shallow ConvNet using encoded data
shallow = get_shallow_net((sec_hidden_size, sec_hidden_size, n_filters))

hist_recon = shallow.fit(img_train, validation_data=img_val, epochs=epochs)

print(
    "Acc: %.3f"
    % (
        sum(hist_recon.history["accuracy"])
        / len(hist_recon.history["accuracy"])
    )
)
print(
    "Val_Acc: %.3f"
    % (
        sum(hist_recon.history["val_accuracy"])
        / len(hist_recon.history["val_accuracy"])
    )
)
print(
    "Loss: %.3f"
    % (sum(hist_recon.history["loss"]) / len(hist_recon.history["loss"]))
)
print(
    "Val_Loss: %.3f"
    % (
        sum(hist_recon.history["val_loss"])
        / len(hist_recon.history["val_loss"])
    )
)

orig_acc = hist_orig.history["val_accuracy"]
recon_acc = hist_recon.history["val_accuracy"]
orig_loss = hist_orig.history["val_loss"]
recon_loss = hist_recon.history["val_loss"]


# Simple verification if metrics of encoded data model were better than original data model
for i, (o_acc, o_loss, r_acc, r_loss) in enumerate(
    zip(orig_acc, orig_loss, recon_acc, recon_loss)
):
    if r_acc >= o_acc:
        print("[!] Atingiu acc na epoch %d" % (i + 1))

    if r_loss <= o_loss:
        print("[!] Atingiu loss na epoch %d" % (i + 1))
