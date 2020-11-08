import tensorflow_probability as tfp

tfd = tfp.distributions

class ToyMC(object):
    def __init__(self,sampling_func):
        self.sampling_func = sampling_func

    def simulate(self,shape,poi):
        return self.sampling_func(shape,poi)

def normal_sampling_func(shape,loc,scale=1.):
    return tfd.Normal(loc=loc,scale=scale).sample(shape)

toymc_normal = normal_sampling_func
