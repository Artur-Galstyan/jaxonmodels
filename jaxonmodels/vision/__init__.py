from jaxtyping import install_import_hook


with install_import_hook(modules=["jaxonmodels"], typechecker="beartype.beartype"):
    import jaxonmodels.vision.resnet  # noqa
