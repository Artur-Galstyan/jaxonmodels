def fn(x: int):
    return x


import inspect

signature = inspect.signature(fn)
params_with_types = {
    name: param.annotation for name, param in signature.parameters.items()
}
print(params_with_types)
