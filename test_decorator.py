import pymc as pm

from pymc_dist_func.normal2 import Normal

# Testing the Normal distribution in a model
with pm.Model() as model:
    foo = Normal("Foo", 2, 5)
    trace = pm.sample()
    print(trace.posterior.Foo.mean())  # should be close to 2
    print(trace.posterior.Foo.std())  # should be close to 2

# Testing the Normal.dist method
bar = Normal.dist(mu=2, sigma=1)
print(bar)
