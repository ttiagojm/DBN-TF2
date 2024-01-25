import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

class CRBM(tf.Module):
    def __init__(self,
                 visible_width,
                 visible_height,
                 visible_channels,
                 kernel_width,
                 kernel_height,
                 kernel_channels,
                 batch_size,
                 maxpooling = True,
                 sparsity_coef = 0.1,
                 sparsity_target = 0.1,
                 gauss_var = 0.2,
                 init_weight_std = 0.01,
                 learning_rate = 0.0001,
                 learning_rate_decay = 0.5,
                 momentum = 0.9,
                 decay_step = 50000,
                 weight_decay = 0.1
                ):

        self.visible_width = visible_width
        self.visible_height = visible_height
        self.visible_channels = visible_channels
        self.kernel_width = kernel_width
        self.kernel_height = kernel_height
        self.kernel_channels = kernel_channels
        self.hidden_height = visible_height # because padding is SAME
        self.hidden_width = visible_width   # because padding is SAME
        self.batch_size = batch_size
        self.maxpooling = maxpooling
        self.sparsity_coef = sparsity_coef
        self.sparsity_target = sparsity_target
        self.gauss_var = gauss_var
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.momentum = momentum
        self.decay_step = decay_step
        self.weight_decay = weight_decay

        # Create variables
        self.kernels = tf.Variable(
            tf.random.truncated_normal(
                [kernel_height, kernel_width, visible_channels, kernel_channels],
                stddev=init_weight_std
            )
        )
        self.v_bias = tf.Variable(tf.constant(0.01, shape=[visible_channels,]), dtype=np.float32)
        self.h_bias = tf.Variable(tf.constant(-3.0, shape=[kernel_channels,]), dtype=np.float32)
        self.sigma = tf.Variable(tf.constant(gauss_var*gauss_var,
                                             shape=[visible_height*visible_width*batch_size*visible_channels,]
                                            ),
                                 dtype=np.float32
                                )

        # Variables to save previous values
        self.kernels_prev = tf.Variable(
            tf.constant(0.0, shape=[kernel_height, kernel_width, visible_channels, kernel_channels]),
            dtype=np.float32
        )
        self.h_bias_prev = tf.Variable(tf.constant(0.0, shape=[kernel_channels,]), dtype=np.float32)
        self.v_bias_prev = tf.Variable(tf.constant(0.0, shape = [visible_channels,]), dtype=np.float32)

        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=self.decay_step,
            decay_rate=self.learning_rate_decay,
            staircase=True
        )

    def energy_function(self, v, h, flow="forward"):

        if flow == "forward":
            conv = tf.nn.conv2d(v, self.kernels, [1, 1, 1, 1], padding='SAME')
            layer = h
        else:
            # Invert kernel to restore input
            kernel_T = tf.transpose(tf.reverse(self.kernels,[True,False]),perm=[0, 1, 3, 2])
            conv = tf.nn.conv2d(h, kernel_T, [1, 1, 1, 1], padding='SAME')
            layer = v

        # Applied element-wise emultiplication followed by sum
        weight =  tf.reduce_sum(tf.multiply(layer,conv))
        h_bias =  tf.reduce_sum(tf.multiply(self.h_bias,tf.reduce_sum(h,[0,1,2])))

        # Apply Z-Score
        weight = tf.div(weight,self.gaussian_variance)
        v_bias = tf.reduce_sum(tf.square(tf.subtract(v,tf.reshape(self.v_bias,[1,1,1,self.visible_channels]))))
        v_bias = tf.divide(v_bias,2*self.gauss_var*self.gauss_var)

        output = tf.subtract(v_bias,tf.add(h_bias,weight))

        return tf.divide(output,self.batch_size)

    def infer_probability(self, layer, flow, result = "hidden"):
        # Returns the new h hidden layer values

        if flow == "forward":
            conv = tf.nn.conv2d(layer, self.kernels, [1, 1, 1, 1], padding='SAME')
            conv = tf.divide(conv,self.gauss_var)
            # Apply bias
            bias = tf.nn.bias_add(conv, self.h_bias)

            if self.maxpooling:
                exp = tf.exp(bias)
                custom_kernel = tf.constant(1.0, shape=[2,2,self.kernel_channels,1])
                summ = tf.nn.depthwise_conv2d(exp, custom_kernel, [1, 2, 2, 1], padding='VALID')
                summ = tf.add(1.0,summ)

                ret_kernel = np.zeros((2,2,self.kernel_channels,self.kernel_channels))

                # Define diagonal with ones
                ret_kernel[:, :, range(self.kernel_channels), range(self.kernel_channels)] = 1

                custom_kernel_bis = tf.constant(ret_kernel,dtype = tf.float32)

                sum_bis =  tf.nn.conv2d_transpose(
                    summ,
                    custom_kernel_bis,
                    (self.batch_size,self.hidden_height,self.hidden_width,self.kernel_channels),
                    strides= [1, 2, 2, 1],
                    padding='VALID'
                )

                if result == "hidden":
                    return tf.divide(exp, sum_bis)

                return tf.subtract(1.0,tf.divide(1.0, summ))

            return tf.sigmoid(bias)

        # Backwards
        kernel_T = tf.transpose(tf.reverse(self.kernels,[True, False]),perm=[0, 1, 3, 2])
        conv = tf.nn.conv2d(layer, kernel_T, [1, 1, 1, 1], padding='SAME')
        conv = tf.multiply(conv,self.gauss_var)

        return tf.nn.bias_add(conv, self.v_bias)

    def draw_samples(self, mean_activation, flow):

        if flow == "forward":
            return tf.where(
                tf.random.uniform([self.batch_size,self.hidden_height,self.hidden_width,self.kernel_channels]) - mean_activation < 0,
                tf.ones([self.batch_size,self.hidden_height,self.hidden_width, self.kernel_channels]),
                tf.zeros([self.batch_size,self.hidden_height,self.hidden_width, self.kernel_channels])
            )

        mean = tf.reshape(mean_activation, [-1])
        dist = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=self.sigma)
        return tf.reshape(
            dist.sample(),
            [self.batch_size,self.visible_height,self.visible_width,self.visible_channels]
        )

    def gibbs_sampling(self, data, num_gibbs = 1):

        v0 = data
        h0 = self.infer_probability(data,'forward')
        ret = self.draw_samples(h0, 'forward')

        for i in range(num_gibbs):
            vn = self.draw_samples(self.infer_probability(ret,'backward'), 'backward')
            hn = self.infer_probability(vn,'forward')
            ret = self.draw_samples(hn, 'forward')
        return v0,h0,vn,hn

    def do_contrastive_divergence(self, data, num_gibbs = 1, global_step = 0):
        v0,h0,vn,hn = self.gibbs_sampling(data, num_gibbs)

        # Padding visible to after conv keep the same size
        v_pad = tf.pad(
            tf.transpose(v0,perm=[3,1,2,0]),
            [[0,0],
             [np.floor((self.kernel_height-1)/2).astype(int),np.ceil((self.kernel_height-1)/2).astype(int)],
             [np.floor((self.kernel_width-1)/2).astype(int),np.ceil((self.kernel_width-1)/2).astype(int)],
             [0,0]]
        )

        positive = tf.nn.conv2d(
            v_pad,
            tf.transpose(h0,perm=[1,2,0,3]), [1, 1, 1, 1], padding='VALID'
        )

        v_pad = tf.pad(
            tf.transpose(vn,perm=[3,1,2,0]),
            [[0,0],
             [np.floor((self.kernel_height-1)/2).astype(int),np.ceil((self.kernel_height-1)/2).astype(int)],
             [np.floor((self.kernel_width-1)/2).astype(int),np.ceil((self.kernel_width-1)/2).astype(int)],
             [0,0]]
        )

        negative = tf.nn.conv2d(
            v_pad,
            tf.transpose(hn,perm=[1,2,0,3]), [1, 1, 1, 1], padding='VALID'
        )

        ret = tf.divide(positive - negative, self.gauss_var)

        g_weight = tf.divide(tf.transpose(ret,perm=[1,2,0,3]),self.batch_size)
        g_weight_sparsity = self.get_sparsity_penalty('kernel', h0, v0)
        g_weight_l2 = tf.multiply(self.weight_decay,self.kernels)

        g_v_bias = tf.divide(tf.reduce_sum(tf.subtract(v0,vn), [0,1,2]),self.batch_size)
        g_v_bias = tf.divide(g_v_bias, self.gauss_var * self.gauss_var)

        g_h_bias = tf.divide(tf.reduce_sum(tf.subtract(h0,hn), [0,1,2]),self.batch_size)
        g_h_bias_sparsity = self.get_sparsity_penalty('hidden_bias', h0, v0)

        ret_w  = self.apply_grad(self.kernels, g_weight, self.kernels_prev, wd = True, wd_value = g_weight_l2, sparsity = True, sparsity_value = g_weight_sparsity,  global_step = global_step)
        ret_bv = self.apply_grad(self.v_bias, g_v_bias, self.v_bias_prev, global_step = global_step)
        ret_bh = self.apply_grad(self.h_bias, g_h_bias, self.h_bias_prev, sparsity = True, sparsity_value = g_h_bias_sparsity, global_step = global_step)
        cost = tf.reduce_sum(tf.square(tf.subtract(data,vn)))
        update = tf.reduce_sum(vn)

        return ret_w, ret_bv, ret_bh, cost, update


    def apply_grad(self, parameter, grad_value, prev_value, wd = False, wd_value = None, sparsity = False, sparsity_value = None, global_step = 0):
        lr = self.lr_schedule(global_step)
        ret = tf.add(tf.multiply(self.momentum,prev_value),tf.multiply(lr,grad_value))
        ret = prev_value.assign(ret)

        if wd:
            ret = tf.subtract(ret,tf.multiply(lr,wd_value))

        if sparsity:
            ret = tf.subtract(ret,tf.multiply(lr,sparsity_value))

        return parameter.assign_add(ret)


    def get_sparsity_penalty(self, name_type, h0, v0):
        ret = -2*self.sparsity_coef/self.batch_size
        ret = ret/self.gauss_var

        mean = tf.reduce_sum(h0, [0], keepdims = True)
        baseline = tf.multiply(
            tf.subtract(self.sparsity_target, mean),
            tf.multiply(tf.subtract(1.0,h0),h0)
        )

        if name_type == 'hidden_bias':
            return tf.multiply(ret,tf.reduce_sum(baseline, [0,1,2]))

        if name_type == 'kernel':
            v_pad = tf.pad(
            tf.transpose(v0,perm=[3,1,2,0]),
            [[0,0],
             [np.floor((self.kernel_height-1)/2).astype(int),np.ceil((self.kernel_height-1)/2).astype(int)],
             [np.floor((self.kernel_width-1)/2).astype(int),np.ceil((self.kernel_width-1)/2).astype(int)],
             [0,0]]
        )
            retBis = tf.nn.conv2d(
                v_pad,
                tf.transpose(baseline,perm=[1,2,0,3]), [1, 1, 1, 1], padding='VALID'
            )
            retBis = tf.transpose(retBis,perm=[1,2,0,3])

        return tf.multiply(ret,retBis)