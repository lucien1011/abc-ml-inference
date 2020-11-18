import numpy as np
import tensorflow as tf
from tensorflow.python.ops.histogram_ops import histogram_fixed_width_bins

class Histogram(tf.keras.layers.Layer):
    def __init__(self,nbins,value_range,input_axis):
        super(Histogram, self).__init__()
        self.nbins = nbins
        self.value_range = value_range
        self.input_axis = input_axis

    def build(self, input_shape):
        pass

    def call(self, input):
        if self.input_axis is None:
            return tf.histogram_fixed_width(input, self.value_range, nbins=self.nbins)
        else:
            if not hasattr(self.input_axis, "__len__"):
                axis = [self.input_axis]
        
            other_axis = [x for x in range(0, len(input.shape)) if x not in axis]
            swap = tf.transpose(input, [*other_axis, *axis])
            flat = tf.reshape(swap, [-1, *np.take(input.shape.as_list(), axis)])
        
            count = tf.map_fn(lambda x: tf.histogram_fixed_width(x, self.value_range, nbins=self.nbins), flat, dtype=(tf.int32))
        
            return tf.reshape(count/input.shape[self.input_axis], [*np.take([-1 if a is None else a for a in input.shape.as_list()], other_axis), self.nbins])
