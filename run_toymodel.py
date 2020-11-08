import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np

import matplotlib.pyplot as plt

from toymc import toymc_normal

from mdn import MDN

tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras
tfkl = tf.keras.layers

mc = toymc_normal

ncomp = 16
nparam = 2
plot_low = -10.
plot_high = 10.
nbin = 512
R = 100
N = 512
saved_model_path = 'saved_model/mdn_201108_01'

model = MDN(nbin,ncomp,nparam)
optimizer = tf.keras.optimizers.Adam()

bins = [plot_low+ibin*(plot_high-plot_low)/nbin for ibin in range(nbin+1)]
for r in range(R):
    means = tfd.Uniform(low=-1., high=1.).sample([N,])
    sigmas = tfd.Uniform(low=0., high=1.).sample([N,])
    x = tf.transpose(toymc_normal((5000,),means,sigmas))
    hists = np.apply_along_axis(lambda x: np.histogram(x, bins=bins, density=1.)[0], 1, x)

    pois = tf.concat([tf.expand_dims(means,axis=1),tf.expand_dims(sigmas,axis=1),],axis=1)
    with tf.GradientTape() as tape:
        inputs = model(hists)
        ll = tf.math.abs(tf.math.log(model.calculate_loss(inputs,pois)))
        ll = tf.reduce_mean(ll)
    if r % 10 == 0: print("Epoch ",r,ll)
    grad = tape.gradient(ll,model.trainable_weights)
    optimizer.apply_gradients(zip(grad,model.trainable_weights))
model.save(saved_model_path)
