import os

import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np

import matplotlib.pyplot as plt

from model.MDN import MDN
from mc.toymodel.NormalGenerator import NormalGenerator
from utils.mkdir_p import mkdir_p

tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras
tfkl = tf.keras.layers

# _________________________________________________________________ ||
# Configurables
# _________________________________________________________________ ||
nbin = 512
nEpoch = 100
batch_size = 512
sample_size = 5000

nDraw = 20
nContourBins = 40
contour_x_low = -2.
contour_x_high = 2.
contour_y_low = -2.
contour_y_high = 2.

saved_model_path = 'saved_model/mdn_201116_03'

# _________________________________________________________________ ||
# Define TF model and optimizer
# _________________________________________________________________ ||
ncomp = 4
nparam = 2
model = MDN(nbin,ncomp,nparam)
optimizer = tf.keras.optimizers.Adam()

# _________________________________________________________________ ||
# Define MC generator
# _________________________________________________________________ ||
plot_low = -10.
plot_high = 10.
generator = NormalGenerator(
        mean_low = -1,
        mean_high = 1.,
        sigma_low = -1.,
        sigma_high = 1.,
        bins = [plot_low+ibin*(plot_high-plot_low)/nbin for ibin in range(nbin+1)],
        )

# _________________________________________________________________ ||
# Training
# _________________________________________________________________ ||
for i in range(nEpoch):
    x,hists,pois,_,_ = generator.generate(batch_size,(sample_size,))
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

bins_mean = [contour_x_low+ibin*(contour_x_high-contour_x_low)/nContourBins for ibin in range(nContourBins+1)]
bins_sigma = [contour_y_low+ibin*(contour_y_high-contour_y_low)/nContourBins for ibin in range(nContourBins+1)]

for i in range(nDraw):
    plt.clf()
    x,hists,pois,means,lnsigmas = generator.generate(1,(sample_size,))
    plt.title(label='mean, sigma: '+str(means[0].numpy())+' '+str(lnsigmas[0].numpy()))
    
    inputs = model(hists)

    X, Y = np.meshgrid(bins_mean,bins_sigma)
    ll = np.array([[model.calculate_loss(inputs,tf.constant([[X[j][i],Y[j][i]]],dtype=np.float32))[0] for i in range(nContourBins+1)] for j in range(nContourBins+1)])
    c = plt.contour(X,Y,ll)
    plt.plot(means,lnsigmas,marker='*',color='red')
    plt.clabel(c, inline=1, fontsize=10)
    plt.grid()
    
    plt.savefig(os.path.join(plot_dir,'toymc_'+str(i)+".png"))
