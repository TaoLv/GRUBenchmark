import time

import theano
from theano import tensor as T
import numpy as np

from mkl_gru_seq_wx import GRU_Inference

seq_length = 10
batch_size = 64
input_size = 150
hidden_size = 1024

nbatch = 100
nsamples = nbatch * batch_size

def build_model():
    x = T.ftensor3('inp')
    wx = theano.shared(np.random.rand(3 * hidden_size, input_size).astype(np.float32))
    wh = theano.shared(np.random.rand(3 * hidden_size, hidden_size).astype(np.float32))
    hs = theano.shared(np.zeros((hidden_size, batch_size), np.float32))

    out = GRU_Inference(hid=hidden_size, return_sequences=True, max_len=10)(x, wx, wh, hs)
    model = theano.function(inputs=[x], outputs=out[-1])

    return model


def bench():
    xinput = np.random.rand(seq_length, input_size, batch_size).astype(np.float32)

    f = build_model()

    # theano.printing.pydotprint(f, outfile='gru_inference.png', var_with_name_simple=True)
    # warmup
    f(xinput)

    tic = time.time()
    for i in xrange(0, nbatch):
        f(xinput)
    toc = time.time()

    print "Forward:"
    print "--- %i samples in %s seconds (%f samples/s, %.7f s/sample) ---" % (nsamples, toc - tic, nsamples / (toc - tic), (toc - tic) / nsamples)


if __name__ == '__main__':
    bench()
