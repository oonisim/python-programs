import datetime
import numpy as np

from data.mnist import load_mnist

from common.constant import (
    TYPE_FLOAT,
    TYPE_INT
)
from network.sequential import (
    SequentialNetwork
)
from network.utility import (
    multilayer_network_specification
)


def test_mnist():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

    # Convert to the numpy default type. Numpy cannot set the default float type globally.
    X_train = x_train.astype(TYPE_FLOAT)
    T_train = t_train.astype(TYPE_INT)
    X_test = x_test.astype(TYPE_FLOAT)

    N = X_train.shape[0]
    D = X_train.shape[1]
    M01 = 32
    M02 = 32
    M03 = 16
    M04 = 10
    M = 10

    specification = multilayer_network_specification([D, M01, M02, M03, M04, M])
    multilayer_network = SequentialNetwork.build(
        specification=specification
    )

    MAX_TEST_TIMES = 100

    elapsed = []
    history_recall = []
    for i in range(MAX_TEST_TIMES):
        start = datetime.datetime.now()

        multilayer_network.train(X=X_train, T=T_train)
        recall = np.sum(multilayer_network.predict(X_test) == t_test) / t_test.size
        history_recall.append(recall)

        end = datetime.datetime.now()
        elapsed.append(end - start)

        if not (i % 5):
            print(f"iteration {i:3d} Loss {multilayer_network.L:15f} Recall {recall:7f} avg {np.mean(elapsed)}")