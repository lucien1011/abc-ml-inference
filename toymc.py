import tensorflow_probability as tfp

tfd = tfp.distributions

def normal_sampling_func(shape,loc,scale=1.):
    return tfd.Normal(loc=loc,scale=scale).sample(shape)

toymc_normal = normal_sampling_func
