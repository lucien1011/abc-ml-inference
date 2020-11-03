import tensorflow_probability as tfp

tfd = tfp.distributions

class ToyMC(object):
    def __init__(self,sampling_func):
        self.sampling_func = sampling_func

    def simulate(self,shape,poi):
        return self.sampling_func(shape,poi)

def normal_sampling_func(shape,poi):
    return tfd.Normal(loc=poi, scale=1.).sample(shape)
toymc_normal = ToyMC(normal_sampling_func)
