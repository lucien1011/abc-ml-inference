import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

class GaussianMC(object):
    def generate(self,mean,sigma,sample_size):
        return  tf.transpose(tfd.Normal(loc=mean,scale=sigma).sample(sample_size))

