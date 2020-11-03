import numpy

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_probability as tfp

tfd = tfp.distributions
tfkl = tf.keras.layers

class MDN(tf.keras.Model):
    def __init__(self,nbin,ndf,**kwargs):
        super(MDN,self).__init__(**kwargs)
        self.ndf = ndf
        self.input_layer = tfkl.InputLayer(input_shape=(nbin,))
        self.dense_layer_1 = tfkl.Dense(512,activation='relu')
        self.dense_layer_2 = tfkl.Dense(24,activation='linear')
        self.reshape_layer = tfkl.Reshape((self.ndf,3))

    def call(self,input_tensor,training=False):
        out = self.input_layer(input_tensor)
        out = self.dense_layer_1(out)
        out = self.dense_layer_2(out)
        out = self.reshape_layer(out)
        return out
    
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None,None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            ]
    )
    def calculate_loss(self,inputs,y):
        z_mean = inputs[:,:,0]
        z_sigma = tf.exp(0.5 * inputs[:,:,1])
        z_rho = inputs[:,:,2]
        dist = tfd.Normal(loc=z_mean, scale=z_sigma)
        batch = tf.shape(z_mean)[0]
        y = tf.broadcast_to(
                tf.reshape(y,(batch,1)),
                (batch,self.ndf),
                )
        pdf = tf.reshape(
                dist.cdf(y),
                (batch,self.ndf),
                ) - 0.5
        z_rho = tf.nn.softmax(z_rho,axis=1)
        out = tf.math.multiply(tf.math.abs(pdf),z_rho)
        return tf.reduce_sum(out,axis=1)
