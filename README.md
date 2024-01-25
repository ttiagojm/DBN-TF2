# DBN-TF2
Deep Belief Networks for image reconstruction in Tensorflow 2.

<hr>

## Convolutional Restricted Boltzman Machine
`DBN/crbm.py` file contains the class where all logic and training code was implemented

The code for CRBM was adapted from [this](https://github.com/arthurmeyer/Convolutional_Deep_Belief_Network) repository. However, it's not being used the max pooling output, since we want to keep the spatial dimensions of the images.


## Discrete Restricted Boltzman Machine
`DBN/RBM/rbm.py` file contains the class where all logic and training code was implemented

Is not being used and the codes needs to be adapted for the new implementation of DBN

<hr>