import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np

import matplotlib.pyplot as plt

from toymc import toymc_normal

from mdn import MDNLayer,MDN

tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras
tfkl = tf.keras.layers

mc = toymc_normal

event_shape = [1]
num_components = 8
model = MDN(256,8)
optimizer = tf.keras.optimizers.Adam()

plot_low = -10.
plot_high = 10.
nbin = 256
R = 500
N = 512
bins = [plot_low+ibin*(plot_high-plot_low)/nbin for ibin in range(nbin+1)]
for r in range(R):
    pois = tfd.Uniform(low=-5., high=5.).sample([N,])
    x = tf.transpose(mc.simulate((5000,),pois))
    hists = np.apply_along_axis(lambda x: np.histogram(x, bins=bins, density=1.)[0], 1, x)
    with tf.GradientTape() as tape:
        inputs = model(hists)
        ll = model.calculate_loss(inputs,pois)
        ll = tf.reduce_mean(ll)
        #nll = tf.reduce_mean(-tf.math.log(ll))
    if r % 10 == 0: print("Epoch ",r,ll)
    grad = tape.gradient(ll,model.trainable_weights)
    optimizer.apply_gradients(zip(grad,model.trainable_weights))
model.save('saved_model/mdn_201103_01')
