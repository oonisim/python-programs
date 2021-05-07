"""Network base test cases"""
import cProfile
import logging

from memory_profiler import profile as memory_profile
import numpy as np
import tensorflow as tf

import function.fileio as fileio
from common.constant import (
    TYPE_INT
)
from common.function import (
    sigmoid_cross_entropy_log_loss
)
from layer import (
    Embedding
)
from layer.adapter import (
    Adapter
)
from layer.objective import (
    CrossEntropyLogLoss
)
from layer.preprocessing import (
    EventContext
)
from layer.preprocessing import (
    EventIndexing,
)
from network.sequential import (
    SequentialNetwork
)

Logger = logging.getLogger(__name__)


@memory_profile
def test_word2vec():
    USE_PTB = True
    DEBUG = False
    VALIDATION = True

    TARGET_SIZE = TYPE_INT(1)  # Size of the target event (word)
    CONTEXT_SIZE = TYPE_INT(4)  # Size of the context in which the target event ocuurs.
    WINDOW_SIZE = TARGET_SIZE + CONTEXT_SIZE
    SAMPLE_SIZE = TYPE_INT(5)  # Size of the negative samples
    VECTOR_SIZE = TYPE_INT(20)  # Number of features in the event vector.

    corpus = "To be, or not to be, that is the question that matters"
    _file = "ptb.train.txt"
    if not fileio.Function.is_file(f"~/.keras/datasets/{_file}"):
        path_to_ptb = tf.keras.utils.get_file(
            _file,
            f'https://raw.githubusercontent.com/tomsercu/lstm/master/data/{_file}'
        )
        corpus = fileio.Function.read_file(path_to_ptb)
    else:
        raise RuntimeError("%s does not exist" % _file)

    # --------------------------------------------------------------------------------
    # Logistic Log Loss
    # --------------------------------------------------------------------------------
    loss = CrossEntropyLogLoss(
        name="loss",
        num_nodes=1,  # Logistic log loss
        log_loss_function=sigmoid_cross_entropy_log_loss
    )

    # --------------------------------------------------------------------------------
    # Event indexing
    # --------------------------------------------------------------------------------
    word_indexing = EventIndexing(
        name="word_indexing_on_ptb",
        corpus=corpus
    )
    del corpus

    # --------------------------------------------------------------------------------
    # Event Context
    # --------------------------------------------------------------------------------
    event_context = EventContext(
        name="ev",
        window_size=WINDOW_SIZE,
        event_size=TARGET_SIZE
    )

    # --------------------------------------------------------------------------------
    # Event Embedding
    # --------------------------------------------------------------------------------
    embedding: Embedding = Embedding(
        name="embedding",
        num_nodes=WINDOW_SIZE,
        target_size=TARGET_SIZE,
        context_size=CONTEXT_SIZE,
        negative_sample_size=SAMPLE_SIZE,
        event_vector_size=VECTOR_SIZE,
        dictionary=word_indexing
    )

    # --------------------------------------------------------------------------------
    # Adapter between Embedding and Log Loss
    # --------------------------------------------------------------------------------
    adapter_function = embedding.adapt_function_to_logistic_log_loss(loss=loss)
    adapter_gradient = embedding.adapt_gradient_to_logistic_log_loss()
    adapter: Adapter = Adapter(
        name="adapter",
        num_nodes=TYPE_INT(1),  # Number of output M=1
        function=adapter_function,
        gradient=adapter_gradient
    )

    # --------------------------------------------------------------------------------
    # Network
    # --------------------------------------------------------------------------------
    network = SequentialNetwork(
        name="word2vec",
        num_nodes=1,
        inference_layers=[
            word_indexing,
            event_context,
            embedding,
            adapter
        ],
        objective_layers=[
            loss
        ]
    )

    NUM_SENTENCES = 10
    MAX_ITERATIONS = 10

    def train():
        stream = fileio.Function.file_line_stream(path_to_ptb)
        try:
            while True:
                sentences = np.array(fileio.Function.take(NUM_SENTENCES, stream))
                yield sentences
        except StopIteration:
            pass
        finally:
            stream.close()

    trainer = train()

    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(MAX_ITERATIONS):
        network.train(X=next(trainer), T=np.array([0]))
        print(network.history[-1])

    profiler.disable()
    profiler.print_stats(sort="cumtime")
