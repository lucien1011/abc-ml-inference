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

event_shape = [1]
ncomp = 8
nparam = 1
plot_low = -10.
plot_high = 10.
nbin = 256
R = 200
N = 512
saved_model_path = 'saved_model/mdn_201104_01'

model = MDN(nbin,ncomp,nparam)
optimizer = tf.keras.optimizers.Adam()

bins = [plot_low+ibin*(plot_high-plot_low)/nbin for ibin in range(nbin+1)]
for r in range(R):
    means = tfd.Uniform(low=-5., high=5.).sample([N,])
    x = tf.transpose(mc.simulate((5000,),means))
    hists = np.apply_along_axis(lambda x: np.histogram(x, bins=bins, density=1.)[0], 1, x)
    with tf.GradientTape() as tape:
        inputs = model(hists)
        ll = -tf.math.log(model.calculate_loss(inputs,means))
        ll = tf.reduce_mean(ll)
    if r % 10 == 0: print("Epoch ",r,ll)
    grad = tape.gradient(ll,model.trainable_weights)
    optimizer.apply_gradients(zip(grad,model.trainable_weights))
model.save(saved_model_path)
