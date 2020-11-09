import numpy,math

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_probability as tfp

tfd = tfp.distributions
tfkl = tf.keras.layers

class MDN(tf.keras.Model):
    def __init__(self,nbin,ndf,nparam,**kwargs):
        super(MDN,self).__init__(**kwargs)
        self.ndf = ndf
        self.nparam = nparam
        self.ndiag = self.nparam
        self.nul = int((self.nparam*self.nparam-self.nparam)/2)
        self.ncov = self.ndiag + self.nul
        self.input_layer = tfkl.InputLayer(input_shape=(nbin,))
        self.dense_layer_1 = tfkl.Dense(512,activation='relu')
        self.dense_layer_2 = tfkl.Dense(512,activation='relu')
        self.dense_layer_3 = tfkl.Dense(512,activation='relu')
        self.dense_layer_4 = tfkl.Dense(512,activation='relu')
        self.dense_layer_5 = tfkl.Dense(self.ndf*(self.nparam+self.ncov)+self.ndf,activation='linear')
        self.reshape_layer = tfkl.Reshape((self.ndf,self.nparam+self.ncov+1))

    def call(self,input_tensor,training=False):
        out = self.input_layer(input_tensor)
        out = self.dense_layer_1(out)
        out = self.dense_layer_2(out)
        out = self.dense_layer_3(out)
        out = self.dense_layer_4(out)
        out = self.dense_layer_5(out)
        out = self.reshape_layer(out)
        return out
    
    @tf.function(input_signature=[tf.TensorSpec(shape=(None,None,None), dtype=tf.float32),tf.TensorSpec(shape=(None,None), dtype=tf.float32),])
    def calculate_loss(self,inputs,y):
        mean = inputs[:,:,:self.nparam]
        batch = tf.shape(mean)[0]
        rho = inputs[:,:,self.nparam:self.nparam+1]
    
        x = tf.broadcast_to(tf.expand_dims(y,axis=1),(batch,self.ndf,self.nparam),)
        
        if self.nparam > 1:
            mean = tf.reshape(mean,(batch,self.ndf,self.nparam,))
            rho = tf.reshape(rho,(batch,self.ndf,))
            ul = tfp.math.fill_triangular(inputs[:,:,self.nparam+1:])
            u = tf.linalg.set_diag( ul,tf.exp( - 0.5 * tf.linalg.diag_part(ul) ) )
            pdf = tfd.MultivariateNormalTriL(mean,u)
            pdf = tf.math.abs(pdf.prob(x))
        elif self.nparam == 1:
            mean = tf.reshape(mean,(batch,self.ndf,self.nparam,))
            rho = tf.reshape(rho,(batch,self.ndf,))
            sigma = tf.exp(- 0.5 * tf.reshape(inputs[:,:,self.nparam+1:],(batch,self.ndf,self.nparam)))
            pdf = tfd.Normal(mean,sigma)
            pdf = tf.reshape(tf.math.abs(pdf.prob(x)),(batch,self.ndf))

        rho = tf.nn.softmax(rho,axis=1)
        out = tf.math.multiply(pdf,rho)
        return tf.reduce_sum(out,axis=1)
