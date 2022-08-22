# DBN-TF2
Deep Belief Networks in Tensorflow 2


⚠️ This repository is under construction.

<hr>

## Discrete Restricted Boltzman Machine
`discrete_rbm.py` file contains the class where all logic and training code was implemented
`test_disc_rbm.py` file has the purpose of testing the RBM implementation

By default only an image was used to train and test. Obviously the reconstruction is almost perfect.

![Original Image](images/1_original_img_train.png) | ![Reconstructed Image](images/1_reconstruct_img_train.png)

If you try with more data like 10% of training samples:
```
	...
	split=["train[:10%]"],
	...
```

You'll notice that perfection was lost...


![Original Image](images/10perc_original_img_train.png) | ![Reconstructed Image](images/10perc_reconstruct_img_train.png)

Increase `hidden_units` can help getting better results but nothing noticeable.

<hr>

## Convolutional Restricted Boltzman Machine
`convolutional_rbm.py` file contains the class where all logic and training code was implemented
`test_conv_rbm.py` file has the purpose of testing the RBM implementation.

Using 10% of training dataset for training and 5% for testing, we can see nice results on the reconstructions

![Original Image](images/10perc_original_conv_img_train.png) | ![Reconstructed Image](images/10perc_reconstruct_conv_img_train.png)


## Problems with above models

**They only work with binary images** which means the model only turn on and off pixels (on=white; off=black)

Next step is being able to have Grayscale results!