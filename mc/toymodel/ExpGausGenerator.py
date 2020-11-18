from .BaseGenerator import BaseGenerator

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

class ExpGausGenerator(BaseGenerator):
    def __init__(self,
            lam_low,lam_high,
            mean_low,mean_high,
            sigma_low,sigma_high,
            r_low,r_high,
            ):
        self.mean_low = mean_low
        self.mean_high = mean_high
        self.sigma_low = sigma_low
        self.sigma_high = sigma_high
        self.lam_low = lam_low
        self.lam_high = lam_high
        self.r_low = r_low
        self.r_high = r_high
        self.exp_pdf = tfd.Exponential
        self.gaus_pdf = tfd.Normal

    def generate(self,batch_size,hist_entry_size,*args):
        lams = tfd.Uniform(low=self.lam_low, high=self.lam_high).sample([batch_size,])
        means = tfd.Uniform(low=self.mean_low, high=self.mean_high).sample([batch_size,])
        sigmas = tfd.Uniform(low=self.sigma_low, high=self.sigma_high).sample([batch_size,])
        rs = tfd.Uniform(low=self.r_low, high=self.r_high).sample([batch_size,])
        
        n_exp = tf.cast(((1.-rs)*hist_entry_size),tf.int32)
        n_gaus = hist_entry_size-n_exp

        x = tf.concat([
            tf.concat([
                tf.expand_dims(tf.transpose(self.exp_pdf(lams[i]).sample(n_exp[i])),axis=0),
                tf.expand_dims(tf.transpose(self.gaus_pdf(means[i],sigmas[i]).sample(n_gaus[i])),axis=0),
                ],axis=1)
            for i in range(batch_size)
            ],axis=0)

        pois = tf.concat([
            tf.expand_dims(lams,axis=1),
            tf.expand_dims(means,axis=1),
            tf.expand_dims(sigmas,axis=1),
            tf.expand_dims(rs,axis=1),
            ],axis=1)

        return x,pois
