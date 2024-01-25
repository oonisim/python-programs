from optimizer.sgd import SGD

SCHEMES = {}
OPTIMIZERS = (
    SGD,
)
for __optimizer in OPTIMIZERS:
    if __optimizer:
        SCHEMES[__optimizer.class_id().lower()] = __optimizer

assert SCHEMES
