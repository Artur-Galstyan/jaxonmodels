import equinox

from jaxonmodels.layers.normalization import BatchNorm

batch_norm = BatchNorm(4, "batch")

print(isinstance(batch_norm, equinox.nn.StatefulLayer))
