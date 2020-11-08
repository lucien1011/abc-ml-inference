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

plot_meanlow = -2.
plot_meanhigh = 2.
plot_sigmalow = -2.
plot_sigmahigh = 2.
plot_low = -10.
plot_high = 10.
nbin = 512
nbinp = 40
R = 20
batch_size = 1
saved_model_path = 'saved_model/mdn_201108_01/'
plot_dir = os.path.join(saved_model_path,"plot")

binsp_mean = [plot_meanlow+ibin*(plot_meanhigh-plot_meanlow)/nbinp for ibin in range(nbinp+1)]
binsp_sigma = [plot_sigmalow+ibin*(plot_sigmahigh-plot_sigmalow)/nbinp for ibin in range(nbinp+1)]
bins = [plot_low+ibin*(plot_high-plot_low)/nbin for ibin in range(nbin+1)]
model = tf.keras.models.load_model(saved_model_path)

vec_loss_func = np.vectorize(lambda x: model.calculate_loss(inputs,x))
for r in range(R):
    plt.clf()
    means = tfd.Uniform(low=-1., high=1.).sample([batch_size,])
    sigmas = tfd.Uniform(low=0., high=1.).sample([batch_size,])
    x = tf.transpose(toymc_normal((5000,),means,sigmas))
    plt.title(label='mean, sigma: '+str(means[0].numpy())+' '+str(sigmas[0].numpy()))
    
    hists = np.apply_along_axis(lambda x: np.histogram(x, bins=bins, density=1.)[0], 1, x)
    inputs = model(hists)

    X, Y = np.meshgrid(binsp_mean,binsp_sigma)
    ll = np.array([[model.calculate_loss(inputs,tf.constant([[X[j][i],Y[j][i]]],dtype=np.float32))[0] for i in range(nbinp+1)] for j in range(nbinp+1)])
    c = plt.contour(X,Y,ll)
    plt.plot(means,sigmas,marker='*',color='red')
    plt.clabel(c, inline=1, fontsize=10)
    plt.grid()
    
    plt.savefig(os.path.join(plot_dir,'toymc_'+str(r)+".png"))
