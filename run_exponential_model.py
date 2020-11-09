import os

import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np

import matplotlib.pyplot as plt

from model.MDN import MDN
from mc.ExponentialGenerator import ExponentialGenerator
from utils.mkdir_p import mkdir_p

tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras
tfkl = tf.keras.layers

# _________________________________________________________________ ||
# Configurables
# _________________________________________________________________ ||
nbin = 512
nEpoch = 50
batch_size = 512
sample_size = 5000

nDraw = 20
nPlotBin = 100
rate_low = 0.
rate_high = 2.

saved_model_path = 'saved_model/mdn_201109_02'

# _________________________________________________________________ ||
# Define TF model and optimizer
# _________________________________________________________________ ||
ncomp = 16
nparam = 1
model = MDN(nbin,ncomp,nparam)
optimizer = tf.keras.optimizers.Adam()

# _________________________________________________________________ ||
# Define MC generator
# _________________________________________________________________ ||
plot_low = -10.
plot_high = 10.
generator = ExponentialGenerator(
        lam_low = 0.,
        lam_high = 2.,
        bins = [plot_low+ibin*(plot_high-plot_low)/nbin for ibin in range(nbin+1)],
        )

# _________________________________________________________________ ||
# Training
# _________________________________________________________________ ||
for i in range(nEpoch):
    x,hists,pois = generator.generate(batch_size,(sample_size,))
    with tf.GradientTape() as tape:
        inputs = model(hists)
        ll = tf.math.abs(tf.math.log(model.calculate_loss(inputs,pois)))
        ll = tf.reduce_mean(ll)
    if i % 10 == 0: print("Epoch ",i,ll)
    grad = tape.gradient(ll,model.trainable_weights)
    optimizer.apply_gradients(zip(grad,model.trainable_weights))

# _________________________________________________________________ ||
# Saving
# _________________________________________________________________ ||
model.save(saved_model_path)

# _________________________________________________________________ ||
# Draw validation contour
# _________________________________________________________________ ||
plot_dir = os.path.join(saved_model_path,"plot")
mkdir_p(plot_dir)
plotBins = [rate_low+ibin*(rate_high-rate_low)/nPlotBin for ibin in range(nPlotBin+1)]
for i in range(nDraw):
    plt.clf()
    x,hists,pois = generator.generate(1,(sample_size,))
    
    inputs = model(hists)
    ll = np.array([ model.calculate_loss(inputs,tf.constant([[b]],dtype=np.float32))[0] for b in plotBins])

    plt.title(label='rate: '+str(pois[0].numpy()))
    plt.arrow(pois[0], 1., 0., -1., head_width=0.05,)
    plt.plot(plotBins,ll)
    plt.grid()
    plt.savefig(os.path.join(plot_dir,'toymc_'+str(i)+".png"))
