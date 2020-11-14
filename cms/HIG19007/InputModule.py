import tensorflow as tf
import numpy as np

from PyLooper.Common.Module import Module

class InputModule(Module):
    def __init__(self,name,scale=35.):
        super(InputModule,self).__init__(name)
        self.scale = scale

    def analyze(self,data,training,cfg):
        cfg.pois = np.array([[m/self.scale] for m in training.data_wrapper.input_path_dict])
        cfg.x = np.array([data[i]["massZ2"].to_numpy()/self.scale for i,m in enumerate(training.data_wrapper.input_path_dict)])
        cfg.inputs = np.apply_along_axis(lambda x: np.histogram(x, bins=cfg.bins, density=1.)[0], 1, cfg.x) 
