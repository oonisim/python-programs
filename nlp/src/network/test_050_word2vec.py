"""Network base test cases"""
import sys
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
    TARGET_SIZE = TYPE_INT(1)   # Size of the target event (word)
    CONTEXT_SIZE = TYPE_INT(4)  # Size of the context in which the target event ocuurs.
    WINDOW_SIZE = TARGET_SIZE + CONTEXT_SIZE
    SAMPLE_SIZE = TYPE_INT(5)   # Size of the negative samples
    VECTOR_SIZE = TYPE_INT(20)  # Number of features in the event vector.

    # --------------------------------------------------------------------------------
    # Corpus text
    # --------------------------------------------------------------------------------
    _file = "ptb.train.txt"
    path_to_corpus = f"~/.keras/datasets/{_file}"
    if fileio.Function.is_file(path_to_corpus):
        pass
    else:
        path_to_corpus = tf.keras.utils.get_file(
            _file,
            f'https://raw.githubusercontent.com/tomsercu/lstm/master/data/{_file}'
        )
    corpus = fileio.Function.read_file(path_to_corpus)

    # path_to_input = "ptb_excerpt.txt"
    path_to_input = path_to_corpus

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

    def sentences_generator(path_to_file, num_sentences):
        stream = fileio.Function.file_line_stream(path_to_file)
        try:
            while True:
                _lines = fileio.Function.take(num_sentences, stream)
                yield np.array(_lines)
        finally:
            stream.close()

    # Restore the state if exists.
    STATE_FILE = "wor2vec_embedding.pkl"
    embedding.load(STATE_FILE)

    # Continue training
    profiler = cProfile.Profile()
    profiler.enable()

    NUM_SENTENCES = 100
    MAX_ITERATIONS = 100000

    total_sentences = 0
    epochs = 0
    source = sentences_generator(
        path_to_file=path_to_input, num_sentences=NUM_SENTENCES
    )
    for i in range(MAX_ITERATIONS):
        try:
            sentences = next(source)
            total_sentences += len(sentences)
            print(sentences[0])
            network.train(X=sentences, T=np.array([0]))
            if i % 5 == 0:
                print(f"Batch {i:05d} of {NUM_SENTENCES} sentences: Loss: {network.history[-1]:15f}")
            if i % 200 == 0:
                print("saving states")
                embedding.save(STATE_FILE)

        except fileio.Function.GenearatorHasNoMore as e:
            # Next epoch
            print(f"epoch {epochs} done")
            epochs += 1
            source.close()
            source = sentences_generator(
                path_to_file=path_to_input, num_sentences=NUM_SENTENCES
            )

        except Exception as e:
            print("Unexpected error:", sys.exc_info()[0])
            source.close()
            raise e

    embedding.save(STATE_FILE)

    profiler.disable()
    profiler.print_stats(sort="cumtime")


if __name__ == '__main__':
    test_word2vec()
