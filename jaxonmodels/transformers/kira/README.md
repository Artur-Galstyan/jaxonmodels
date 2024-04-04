# That's right. I'm _**Kira**_ ✍️


_Kira_ is a suite of LLMs built with JAX and [Equinox](https://github.com/patrick-kidger/equinox).

It is designed to be as clean and simple as possible, while still being 
flexible and powerful. It also provides a simple training loop, which interoperates with 
the [Jaxonloader](https://github.com/Artur-Galstyan/jaxonloader) library.

--- 

## Available Models

Currently, _Kira_ provides the following models:
- `Kira`: A standard transformer, which allows for interpolation between MHA and MQA.
- `Mamba`: The new selective state space model

_Kira_ can also be used as an encoder. Simply pass `mask=None` when you call _Kira_ 
and there will be no masking in the MHA (i.e. making it an encoder).

---

To get started with _Kira_, you can either install it with

```
pip3 install kira_llm
```

or simply clone the repository and cherry-pick what you need.

