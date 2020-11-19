import os,uproot
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

from model.FunctionalMDN import build_model,calculate_loss,sample
from mc.toymodel.NormalGenerator import NormalGenerator
from utils.mkdir_p import mkdir_p

tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras
tfkl = tf.keras.layers

# _________________________________________________________________ ||
# Configurables
# _________________________________________________________________ ||
n_bin = 512
n_iter = 1
n_epoch = 30
batch_size = 512
sample_size = 5000

saved_model_path = 'saved_model/mdn_zboson_201118_02/'

# _________________________________________________________________ ||
# Define prior
# _________________________________________________________________ ||
prior_mean = tfd.Normal(loc=0.,scale=1.)
prior_sigma = tfd.Normal(loc=1.,scale=0.1)

# _________________________________________________________________ ||
# Define MC generator
# _________________________________________________________________ ||
from mc.generator.GaussianMC import GaussianMC
mc_gen = GaussianMC()

# _________________________________________________________________ ||
# Read data
# _________________________________________________________________ ||
rootfile = uproot.open("/Users/lucien/CMS/NTuple/lucien/Higgs/DarkZ-NTuple/20181116/SkimTree_DarkPhoton_ZX_Run2016Data_m4l70/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8.root")
tree = rootfile["passedEvents"]
data = (tree.array("massZ1")-90.)/10.
data = np.expand_dims(data,axis=0)

# _________________________________________________________________ ||
# Define TF model and optimizer
# _________________________________________________________________ ||
ncomp = 4
nparam = 2
model = build_model(n_bin,ncomp,nparam)
optimizer = tf.keras.optimizers.Adam()

# _________________________________________________________________ ||
# Train prior
# _________________________________________________________________ ||
print("-"*100)
print("Training prior")
prior_t_mean = prior_mean
prior_t_sigma = prior_sigma
prop_mean = prior_t_mean.sample(batch_size)
prop_sigma = prior_t_sigma.sample(batch_size)
pois = np.concatenate([np.expand_dims(prop_mean,axis=1),np.expand_dims(prop_sigma,axis=1)],axis=1,)
for i in range(n_epoch):
    x = mc_gen.generate(prop_mean,prop_sigma,sample_size)
    with tf.GradientTape() as tape:
        inputs = model(x)
        ll = tf.math.abs(tf.math.log(
            calculate_loss(inputs,pois,nparam,ncomp))
            )
        ll = tf.reduce_mean(ll)
    if i % 10 == 0: print("Epoch ",i,ll)
    grad = tape.gradient(ll,model.trainable_weights)
    optimizer.apply_gradients(zip(grad,model.trainable_weights))
prior_t = tf.keras.models.clone_model(model)
print("-"*100)

# _________________________________________________________________ ||
# Train model
# _________________________________________________________________ ||
print("-"*100)
print("Training model")
for it in range(n_iter):
    
    print("="*100)
    print("Iteration "+str(it))
    print("="*100)
    
    data_inputs = prior_t(data)
    pois = sample(data_inputs,sample_size,nparam) # shape: (batch_size,nparam)
    prop_mean = pois[:,0]
    prop_sigma = pois[:,1]

    for i in range(n_epoch):

        x = mc_gen.generate(prop_mean,prop_sigma,sample_size)
        with tf.GradientTape() as tape:
            inputs = model(x)
            ll = tf.math.abs(tf.math.log(calculate_loss(inputs,pois,nparam,ncomp)))
            ll *= prior_mean.prob(prop_mean)
            ll *= prior_sigma.prob(prop_sigma) 
            ll /= calculate_loss(tf.broadcast_to(data_inputs,(sample_size,data_inputs.shape[1],data_inputs.shape[2])),pois,nparam,ncomp)
            ll = tf.reduce_mean(ll)
        if i % 10 == 0: print("Epoch ",i,ll)
        grad = tape.gradient(ll,model.trainable_weights)
        optimizer.apply_gradients(zip(grad,model.trainable_weights))

    prior_t = tf.keras.models.clone_model(model)
print("-"*100)

model.save(saved_model_path+"posterior")

# _________________________________________________________________ ||
# Draw posterior
# _________________________________________________________________ ||
n_contour_bins = 40
contour_x_low = -4.
contour_x_high = 4.
contour_y_low = 0.
contour_y_high = 2.

plot_dir = os.path.join(saved_model_path,"posterior")
mkdir_p(plot_dir)

bins_mean = [contour_x_low+ibin*(contour_x_high-contour_x_low)/n_contour_bins for ibin in range(n_contour_bins+1)]
bins_sigma = [contour_y_low+ibin*(contour_y_high-contour_y_low)/n_contour_bins for ibin in range(n_contour_bins+1)]

plt.clf()
inputs = model(data)

X, Y = np.meshgrid(bins_mean,bins_sigma)
ll = np.array([[calculate_loss(inputs,tf.constant([[X[j][i],Y[j][i]]],dtype=np.float32),nparam,ncomp)[0] for i in range(n_contour_bins+1)] for j in range(n_contour_bins+1)])
c = plt.contour(X,Y,ll)
plt.plot(np.mean(data),np.std(data),marker='*',color='red')
plt.clabel(c, inline=1, fontsize=10)
plt.grid()
plt.savefig(os.path.join(plot_dir,'plot2d.png'))
