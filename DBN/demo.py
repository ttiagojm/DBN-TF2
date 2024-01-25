import pickle
import matplotlib.pyplot as plt
import numpy as np
from cdbn import CDBN
from CRBM.crbm import CRBM
from utils import get_dataset, test_net, preprocessing
from sklearn.model_selection import ParameterGrid
from datetime import datetime

BATCH_SIZE = 64
ds_train, ds_test, IMG_SHAPE = get_dataset(BATCH_SIZE)

# Combination of parameters to find the best combination for reconstruction
# It's validated comparing the classification of a CNN with original dataset and with top 1 generated dataset
params = ParameterGrid({
    "crbm1_kernel": [3, 5],
    "crbm1_num_kernel": [32, 64],
    "crbm1_max": [True, False],
    "crbm2_kernel": [3, 5],
    "crbm2_num_kernel": [32, 64],
    "crbm2_max": [True, False],
    "global_step": [1]
})


# Save historic data
hists = dict()

for param in params:
    init = datetime.now()

    cdbn = CDBN(BATCH_SIZE, param["global_step"])

    cdbn.crbms.append(CRBM(
        IMG_SHAPE[1], IMG_SHAPE[0], IMG_SHAPE[2],
        param["crbm1_kernel"], param["crbm1_kernel"],
        param["crbm1_num_kernel"],BATCH_SIZE, param["crbm1_max"]
    ))

    cdbn.crbms.append(CRBM(
        IMG_SHAPE[1], IMG_SHAPE[0], param["crbm1_num_kernel"],
        param["crbm2_kernel"], param["crbm2_kernel"],
        param["crbm2_num_kernel"],BATCH_SIZE, param["crbm2_max"]
    ))


    # Pre-train DBN
    ds = ds_train.map(lambda x, _: x).cache()
    cdbn.pretrain(ds)

    # Test DBN generated dataset
    ds_transformed_train, ds_transformed_test = preprocessing(cdbn, ds_train, ds_test)

    # Plot some original and reconstructed images
    ds_train_img = next(iter(ds))[:6]
    ds_dbn_img = next(iter(ds_transformed_train))[0][:6]
    
    fig, ax = plt.subplots(5, 2)
    
    for a, col in zip(ax[0], ["Original", "Reconstructed"]):
        a.set_title(col)
        
    for i in range(5):
        ax[i][0].imshow(ds_train_img[i])
        ax[i][1].imshow(ds_dbn_img[i])
    plt.show()

    # Historic data
    hists[str(param)] = test_net(ds_transformed_train, ds_transformed_test, IMG_SHAPE)

    print("TIME ELAPSED: ", (datetime.now()-init).total_seconds())

hist = test_net(ds_train, ds_test)
hists["original"] = hist


# Serialize hist data
with open("hist.pickle", "wb") as outfile:
    pickle.dump(hists, outfile)

# Deserialize hist data
with open("hist.pickle", "rb") as infile:
    hists_recons = pickle.load(infile)

# Get results from original dataset
original = hists_recons.pop("original")

# Compare top and bottom 3 results
plt.style.use('ggplot')
keys = sorted(hists_recons, 
              key=lambda k: sum(hists_recons[k].history["val_accuracy"])/len(hists_recons[k].history["val_accuracy"]), 
              reverse=True)

for metric in ["accuracy", "val_accuracy", "loss", "val_loss"]:
    best_line = "--"
    for k in keys[:3]:
        acc = hists_recons[k].history[metric]
        plt.plot(np.arange(1, len(acc)+1), acc, label=k, linestyle=best_line)
        best_line = None

    for k in keys[-3:]:
        acc = hists_recons[k].history[metric]
        plt.plot(np.arange(1, len(acc)+1), acc, label=k)

    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', borderaxespad=0.)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.show()
    plt.clf()


# Compare top 1 DBN config generated dataset and original dataset
plt.style.use('ggplot')
for metric in ["accuracy", "val_accuracy", "loss", "val_loss"]:
    best_acc = hists_recons[keys[0]].history[metric]
    orig_acc = original.history["epoch_times"]

    plt.plot(np.arange(1, len(best_acc)+1), best_acc, label=keys[0])
    plt.plot(np.arange(1, len(orig_acc)+1), orig_acc, label="Original")
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', borderaxespad=0.)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.show()
    plt.clf()

# Compare the time spent with top 1 DBN config generated dataset and original dataset
plt.style.use('ggplot')
best = hists_recons[keys[0]].history["epoch_times"]
orig = original.history["epoch_times"]

plt.plot(np.arange(1, len(best)+1), best, label="Top 1")
plt.plot(np.arange(1, len(orig)+1), orig, label="Original")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Time (in seconds)")

plt.style.use('ggplot')
best = hists_recons[keys[0]].history["epoch_times"]
orig = original.history["epoch_times"]
plt.boxplot([best, orig], labels=["Top 1", "Original"])
plt.xlabel("Dataset")
plt.ylabel("Time (in seconds)")