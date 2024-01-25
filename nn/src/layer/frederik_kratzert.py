"""
http://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
"""
import numpy as np


def batchnorm_forward(x, gamma, beta, eps):
    N, D = x.shape
    ddof = 1 if N > 1 else 0

    # step1: calculate mean
    mu = 1. / N * np.sum(x, axis=0)

    # step2: subtract mean vector of every trainings example
    xmu = x - mu

    # step3: following the lower branch - calculation denominator
    sq = xmu ** 2

    # step4: calculate variance
    var = 1. / (N-ddof) * np.sum(sq, axis=0)

    # step5: add eps for numerical stability, then sqrt
    if eps == 0.0:
        sd = np.sqrt(var)
        sd[sd == 0] = 1.0
    else:
        sd = np.sqrt(var + eps)

    # step6: invert sqrtwar
    norm = 1. / sd

    # step7: execute normalization
    xhat = xmu * norm

    # step8: Nor the two transformation steps
    gammax = gamma * xhat

    # step9
    out = gammax + beta

    # store intermediate
    cache = (xhat, gamma, xmu, norm, sd, var, eps)

    return out, cache


def batchnorm_backward(dout, cache):
    # unfold the variables stored in cache
    xhat, gamma, xmu, norm, sd, var, eps = cache

    # get the dimensions of the input/output
    N, D = dout.shape
    ddof = 1 if N > 1 else 0

    # step9
    dbeta = np.sum(dout, axis=0)

    # step8
    dgamma = np.sum(dout * xhat, axis=0)
    dxhat = dout * gamma

    # step7
    dnorm = np.sum(dxhat * xmu, axis=0)
    dxmu1 = dxhat * norm

    # step6
    dsd = dnorm * -1. / (sd ** 2)

    # step5
    dvar = 0.5 * 1. / sd * dsd

    # step4
    dsq = 1. / N * np.ones((N, D)) * dvar

    # step3
    dxmu2 = 2 * xmu * dsq

    # step2
    dx1 = (dxmu1 + dxmu2)
    dmu = -1 * np.sum(dxmu1 + dxmu2, axis=0)

    # step1
    dx2 = 1. / N * np.ones((N, D)) * dmu

    # step0
    dx = dx1 + dx2

    return dx, dgamma, dbeta, dxhat, dvar, dxmu2, dxmu1, dmu
