# coding: utf-8
import sys
sys.path.append('..')
from rnnlm import Rnnlm
from dataset import ptb
from src.common import eval_perplexity


if __name__ == '__main__':
    model = Rnnlm()
    #model = BetterRnnlm()

    # 学習済みのパラメータの読み込み
    model.load_params()

    corpus, _, _ = ptb.load_data('test')

    model.reset_state()
    ppl_test = eval_perplexity(model, corpus)
    print('test perplexity: ', ppl_test)
