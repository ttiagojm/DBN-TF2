import tensorflow as tf
from utils import get_datasets
from dbn import DBN
from crbm import RBMConv
"""
	An example to compare a shallow convnet trained with original and autoencoded images 
"""

# Get datasets
img_train, img_val, img_test, ds_info = get_datasets("fashion_mnist")

in_size = ds_info.features["image"].shape


# Be careful with this value.
# It will be used sequentially in each RBM, e.g knowing that in_size[0] = 28 and dec_val = 6:
# RBM 1 -> hidden_units = in_size[0] - dec_val = 22
# RBM 2 -> RBM_1_hidden_units - dec_val = 16
# We can conclude that dec_val < in_size[0] // 2, in case that we have only 2 RBMs
dec_val = 4


print("#### Model trained with original images ####\n")


## Create a shallow ConvNet using normal and resized data
shallow = tf.keras.Sequential([
    tf.keras.Input(shape=in_size),
    tf.keras.layers.Conv2D(32, (3,3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3,3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.GlobalMaxPooling2D(),
    tf.keras.layers.Dense(10)
])

shallow.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

hist = shallow.fit(img_train, validation_data=img_val, epochs=10)

print( "Acc: %.3f" % (sum(hist.history["accuracy"]) / len(hist.history["accuracy"])) )
print( "Val_Acc: %.3f" % (sum(hist.history["val_accuracy"]) / len(hist.history["val_accuracy"])) )
print( "Loss: %.3f" % (sum(hist.history["loss"]) / len(hist.history["loss"])) )
print( "Val_Loss: %.3f" % (sum(hist.history["val_loss"]) / len(hist.history["val_loss"])) )


print("#### Model trained with reconstructed images ####\n")


model = tf.keras.Sequential([
    DBN([
            RBMConv(in_size[0] - dec_val, 8, lr=.1),
            RBMConv(in_size[0] - dec_val*2, 8, lr=.1)
        ])
    ])

model.layers[0].fit(img_train)

## Create a shallow ConvNet using normal and resized data
shallow = tf.keras.Sequential([
	tf.keras.Input(shape=in_size),
    model,
    tf.keras.layers.Conv2D(32, (3,3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3,3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.GlobalMaxPooling2D(),
    tf.keras.layers.Dense(10)
])

shallow.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

hist = shallow.fit(img_train, validation_data=img_val, epochs=10)

print( "Acc: %.3f" % (sum(hist.history["accuracy"]) / len(hist.history["accuracy"])) )
print( "Val_Acc: %.3f" % (sum(hist.history["val_accuracy"]) / len(hist.history["val_accuracy"])) )
print( "Loss: %.3f" % (sum(hist.history["loss"]) / len(hist.history["loss"])) )
print( "Val_Loss: %.3f" % (sum(hist.history["val_loss"]) / len(hist.history["val_loss"])) )