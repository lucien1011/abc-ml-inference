from .BaseGenerator import BaseGenerator

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

class NormalGenerator(BaseGenerator):
    def __init__(self,mean_low,mean_high,sigma_low,sigma_high,bins):
        self.mean_low = mean_low
        self.mean_high = mean_high
        self.sigma_low = sigma_low
        self.sigma_high = sigma_high
        self.bins = bins
        self.pdf = tfd.Normal

    def generate(self,batch_size,hist_shape,*args):
        means = tfd.Uniform(low=self.mean_low, high=self.mean_high).sample([batch_size,])
        lnsigmas = tfd.Uniform(low=self.sigma_low, high=self.sigma_high).sample([batch_size,])
        sigmas = tf.math.exp(lnsigmas)
        x = tf.transpose(self.pdf(loc=means,scale=sigmas).sample(hist_shape))
        hists = np.apply_along_axis(lambda x: np.histogram(x, bins=self.bins, density=1.)[0], 1, x)
        pois = tf.concat([tf.expand_dims(means,axis=1),tf.expand_dims(lnsigmas,axis=1),],axis=1)
        return x,hists,pois,means,lnsigmas
