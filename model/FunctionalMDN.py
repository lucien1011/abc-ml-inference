import numpy,math

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_probability as tfp

from .Histogram import Histogram

tfd = tfp.distributions
tfkl = tf.keras.layers

def build_model(nbin,ndf,nparam):
    ndiag = nparam
    nul = int((nparam*nparam-nparam)/2)
    ncov = ndiag + nul
    input_layer = tfkl.Input(shape=(nbin,))
    histogram_layer_1 = Histogram(500,[0.,1.],1)
    dense_layer_1 = tfkl.Dense(512,activation='relu')
    dense_layer_2 = tfkl.Dense(512,activation='relu')
    dense_layer_3 = tfkl.Dense(512,activation='relu')
    dense_layer_4 = tfkl.Dense(512,activation='relu')
    dense_layer_5 = tfkl.Dense(ndf*(nparam+ncov)+ndf,activation='linear')
    reshape_layer = tfkl.Reshape((ndf,nparam+ncov+1))
    epsilon = 1E-7
    
    out = input_layer
    out = histogram_layer_1(out)
    out = dense_layer_1(out)
    out = dense_layer_2(out)
    out = dense_layer_3(out)
    out = dense_layer_4(out)
    out = dense_layer_5(out)
    out = reshape_layer(out)
    
    model = tf.keras.models.Model(input_layer,out)
    return model

@tf.function(input_signature=[
    tf.TensorSpec(shape=(None,None,None), dtype=tf.float32),
    tf.TensorSpec(shape=(None,None), dtype=tf.float32),
    tf.TensorSpec(shape=(), dtype=tf.int32),
    tf.TensorSpec(shape=(), dtype=tf.int32),
    tf.TensorSpec(shape=(), dtype=tf.float32),
    ]
    )
def calculate_loss(inputs,y,nparam,ndf,epsilon=1E-7):
    mean = inputs[:,:,:nparam]
    batch = tf.shape(mean)[0]
    rho = inputs[:,:,nparam:nparam+1]

    x = tf.broadcast_to(tf.expand_dims(y,axis=1),(batch,ndf,nparam),)
    
    if nparam > 1:
        mean = tf.reshape(mean,(batch,ndf,nparam,))
        rho = tf.reshape(rho,(batch,ndf,))
        ul = tfp.math.fill_triangular(inputs[:,:,nparam+1:])
        u = tf.linalg.set_diag( ul,tf.exp( - 0.5 * tf.linalg.diag_part(ul) ) )
        pdf = tfd.MultivariateNormalTriL(mean,u)
        pdf = tf.math.abs(pdf.prob(x))
    else:
        mean = tf.reshape(mean,(batch,ndf,nparam,))
        rho = tf.reshape(rho,(batch,ndf,))
        sigma = tf.exp(- 0.5 * tf.reshape(inputs[:,:,nparam+1:],(batch,ndf,nparam)))
        pdf = tfd.Normal(mean,sigma)
        pdf = tf.reshape(tf.math.abs(pdf.prob(x)),(batch,ndf))

    rho = tf.nn.softmax(rho,axis=1)
    out = tf.math.multiply(pdf,rho)
    return tf.reduce_sum(out,axis=1)+epsilon

def sample(inputs,sample_size,nparam):
    rho = tf.squeeze(tf.nn.softmax(inputs[:,:,nparam:nparam+1],axis=1),axis=2)
    cat_samples = tf.random.categorical(tf.math.log(rho),1)
    mean = tf.squeeze(tf.gather(inputs[:,:,:nparam],cat_samples,axis=1,batch_dims=1))
    ul = tf.squeeze(tf.gather(inputs[:,:,:nparam+1],cat_samples,axis=1,batch_dims=1))
    ul = tfp.math.fill_triangular(ul)
    u = tf.linalg.set_diag( ul,tf.exp( - 0.5 * tf.linalg.diag_part(ul) ) )
    pdf = tfp.distributions.MultivariateNormalTriL(mean,u)
    samples = pdf.sample(sample_size)
    return samples
