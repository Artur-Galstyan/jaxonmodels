# Jaxonmodels

This library consists of deep learning model implementations in JAX using Equinox as the neural network library.

The goal of this library is to provide simple, yet performant and easy to understand implementations with the aim to give *exactly* the same output as their Pytorch counterparts. As such, great emphasis is placed on making sure that the layers and the models behave accordingly.

Using `statedict2pytree` we can also load the Pytorch model weights into the JAX models.

Some models will have inadvertently repeated code, but this is fine so long as the model remains self contained for the most part.

## Implemented Models

These models have been implemented:
- [x] AlexNet
- [x] CLIP
- [x] EfficientNet
- [x] ResNet
- [x] ViT
- [x] Mamba
- [x] ConvNext
- [ ] Swin Transformer (in progress)


## Contributing

If you have a model that you would like to include, then just open up a PR. It should contain your model and ideally a few tests showcasing that the model (and its components) behave like their Pytorch versions.
