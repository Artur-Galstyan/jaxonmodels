import jax

from jaxonmodels.models.resnet import resnet18
from jaxonmodels.statedict2pytree.s2p import pytree_to_fields

r, s = resnet18(key=jax.random.key(0))
fields, _ = pytree_to_fields((r, s))

for f in fields:
    print(jax.tree_util.keystr(f.path))
