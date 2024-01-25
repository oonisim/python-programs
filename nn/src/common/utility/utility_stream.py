def generator(func):
    def start(*args, **kwargs):
        g = func(*args, **kwargs)
        # next(g) is the same but be clear intention of advancing the execution to the yield line.
        g.send(None)
        return g

    return start
