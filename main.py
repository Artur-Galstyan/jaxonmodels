import jax

from jaxonmodels.models.resnet import load_resnet

resnet, state = load_resnet("resnet18")
leaves, treedef = jax.tree_util.tree_flatten(resnet)

print(type(resnet.n_classes))

# assert leaves == ["normal"]
# assert "static" in str(treedef)
# treedef
# print(str(treedef))
# print(leaves)
