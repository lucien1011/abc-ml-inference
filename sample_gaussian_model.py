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
nbin        = 512
batch_size  = 5000
sample_size = 5000
nparam      = 2
n_plot      = 20

saved_model_path = 'saved_model/mdn_201116_03'

# _________________________________________________________________ ||
# Define TF model and optimizer
# _________________________________________________________________ ||
model = tf.keras.models.load_model(saved_model_path)

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
# MCMC
# _________________________________________________________________ ||
plot_dir = os.path.join(saved_model_path,"plot_sampling/")
for iplot in range(n_plot):
    plt.clf()
    x,hists,pois,_,_ = generator.generate(1,(sample_size,))
    hists = tf.broadcast_to(hists,(batch_size,hists.shape[1]))
    inputs = model(hists)
    
    print("Sampling pdf")
    rho = tf.squeeze(tf.nn.softmax(inputs[:,:,nparam:nparam+1],axis=1))
    cat_samples = tf.random.categorical(tf.math.log(rho),1)
    mean = tf.squeeze(tf.gather(inputs[:,:,:nparam],cat_samples,axis=1,batch_dims=1))
    ul = tf.squeeze(tf.gather(inputs[:,:,:nparam+1],cat_samples,axis=1,batch_dims=1))
    ul = tfp.math.fill_triangular(ul)
    u = tf.linalg.set_diag( ul,tf.exp( - 0.5 * tf.linalg.diag_part(ul) ) )
    pdf = tfp.distributions.MultivariateNormalTriL(mean,u)
    samples = pdf.sample()
    
    plot_x_array = samples[:,0].numpy()
    plot_y_array = samples[:,1].numpy()
    counts, xedges, yedges, im = plt.hist2d(plot_x_array,plot_y_array,bins=50,range=[[-2.,2.],[-2.,2.],])
    plt.colorbar(im)
    plt.title("Number of samples: "+str(batch_size))
    plt.plot(pois[:,0],pois[:,1],marker='*',color='red')
    plt.savefig(os.path.join(plot_dir,"sampling_"+str(iplot)+".png"))
    print("-"*100)
