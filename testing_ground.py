import equinox as eqx
import jax.numpy as jnp


class InferenceableModule(eqx.Module):
    # inference: bool = eqx.field(static=True)
    inference: bool

    cool: bool = eqx.field(static=True)

    def __init__(self, inference: bool = False):
        self.inference = inference
        self.cool = True

    def __call__(self, x):
        if self.inference:
            return x + 1
        else:
            return x - 1


model = InferenceableModule(inference=False)
x = jnp.ones(shape=(1, 1))

print(model(x))

inf_model = eqx.nn.inference_mode(model)
print(inf_model(x))
