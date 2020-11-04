import os

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_probability as tfp

import numpy as np

import matplotlib.pyplot as plt

from toymc import toymc_normal

tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras
tfkl = tf.keras.layers

mc = toymc_normal

plot_low = -10.
plot_high = 10.
nbin = 256
R = 20
batch_size = 1
saved_model_path = 'saved_model/mdn_201104_01/'
plot_dir = os.path.join(saved_model_path,"plot")

bins = [plot_low+ibin*(plot_high-plot_low)/nbin for ibin in range(nbin+1)]
model = tf.keras.models.load_model(saved_model_path)

for r in range(R):
    plt.clf()
    pois = tfd.Uniform(low=-5., high=5.).sample([batch_size,])
    x = tf.transpose(mc.simulate((5000,),pois))
    plt.hist(x,bins,density=1.,)
    plt.title(label='mean: '+str(pois[0]))
    
    hists = np.apply_along_axis(lambda x: np.histogram(x, bins=bins, density=1.)[0], 1, x)
    inputs = model.predict(hists)

    ll = [tf.reshape(model.calculate_loss(inputs,tf.constant([b])),(1,)) for b in bins]
    plt.plot(bins,ll)
    
    plt.savefig(os.path.join(plot_dir,'toymc_'+str(r)+".png"))
