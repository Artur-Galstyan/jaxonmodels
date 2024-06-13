import json

import jax
from jaxonmodels.transformers.llama.llama3 import LLaMA
from jaxonmodels.transformers.llama.model_args import LLaMAModelArgs


with open("params.json", "r") as f:
    params = json.load(f)

model_args = LLaMAModelArgs(**params)
key = jax.random.key(2)

model = LLaMA(model_args, key=key)
