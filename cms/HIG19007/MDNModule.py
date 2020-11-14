import os
import tensorflow as tf

from PyLooper.Common.Module import Module

from model.MDN import MDN

class MDNModule(Module):
    def __init__(self,nfeature,ncomp,nparam,):
        self.nfeature = nfeature
        self.ncomp = ncomp
        self.nparam = nparam

    def begin(self,training,cfg):
        training.model = MDN(self.nfeature,self.ncomp,self.nparam)
        self.optimizer = tf.keras.optimizers.Adam()

    def analyze(self,data,training,cfg):
        with tf.GradientTape() as tape:
            pred = training.model(cfg.inputs)
            ll = tf.math.abs(tf.math.log(training.model.calculate_loss(pred,cfg.pois)))
            ll = tf.reduce_mean(ll)
        grad = tape.gradient(ll,training.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grad,training.model.trainable_weights))
        training.report.misc_str = "Loss: {0:6.2f}".format(ll)

    def end(self,training,cfg):
        training.model.save(os.path.join(cfg.collector.output_path,training.name))
