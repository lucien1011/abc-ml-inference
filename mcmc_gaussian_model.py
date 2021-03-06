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
batch_size = 1
sample_size = 5000

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
num_chains = 1000
num_results = 300
num_plot = 10
plot_dir = os.path.join(saved_model_path,"plot_sampling/")

for iplot in range(num_plot):
    print("-"*100)
    print("Draw plot "+str(iplot))

    plt.clf()

    x,hists,pois,_,_ = generator.generate(batch_size,(sample_size,))
    inputs = model(hists)

    #nparam = df_x.shape[1]
    #rho = tf.squeeze(inputs[:,:,nparam:nparam+1])
    #dist = tfp.distributions.Categorical(probs=rho)
    #sample = dist.sample(num_chains)
    #mean = tf.reshape(inputs[ibatch,sample,:nparam],(cfg.nparam))
    #ul = tfp.math.fill_triangular(inputs[ibatch,sample,cfg.nparam+1:])
    #u = tf.linalg.set_diag( ul,tf.exp( - 0.5 * tf.linalg.diag_part(ul) ) )
    #u = tf.reshape(u,(cfg.nparam,cfg.nparam))
    #pdf = tfp.distributions.MultivariateNormalTriL(mean,u)
    #sample = pdf.sample(1)
    
    init_state = np.random.normal(-1.,1.,(num_chains, 2)).astype(np.float32)
    inputs = tf.broadcast_to(inputs,(num_chains,inputs.shape[1],inputs.shape[2]))
    
    print("Sampling pdf")
    samples = tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=init_state,
        kernel=tfp.mcmc.RandomWalkMetropolis(lambda x: tf.math.log(model.calculate_loss(inputs,x))),
        num_burnin_steps=10,
        num_steps_between_results=1,
        trace_fn=None,
        seed=42,
        )
    
 
    plot_x_array = samples[num_results-1,:,0].numpy()
    plot_y_array = samples[num_results-1,:,1].numpy()
    counts, xedges, yedges, im = plt.hist2d(plot_x_array,plot_y_array,bins=50,range=[[-2.,2.],[-2.,2.],])
    plt.colorbar(im)
    plt.title("Number of samples: "+str(samples.shape[1]))
    plt.plot(pois[:,0],pois[:,1],marker='*',color='red')
    plt.savefig(os.path.join(plot_dir,"plot_"+str(iplot)+".png"))
    print("-"*100)
