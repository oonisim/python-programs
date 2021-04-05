from optimizer.sgd import SGD

SCHEMES = {}
OPTIMIZERS = (
    SGD,
)
for __optimizer in OPTIMIZERS:
    if __optimizer:
        SCHEMES[__optimizer.__qualname__.lower()] = __optimizer

assert SCHEMES
