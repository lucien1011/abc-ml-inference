from .BaseGenerator import BaseGenerator

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

class ExponentialGenerator(BaseGenerator):
    def __init__(self,lam_low,lam_high,bins):
        self.lam_low = lam_low
        self.lam_high = lam_high
        self.bins = bins
        self.pdf = tf.random.poisson

    def generate(self,batch_size,hist_shape,*args):
        lams = tfd.Uniform(low=self.lam_low, high=self.lam_high).sample([batch_size,])
        x = tf.transpose(self.pdf(hist_shape,lams))
        hists = np.apply_along_axis(lambda x: np.histogram(x, bins=self.bins, density=1.)[0], 1, x)
        pois = tf.expand_dims(lams,axis=1)
        return x,hists,pois
