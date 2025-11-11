import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def self_attention(x, Wq, Wk, Wv):
    d = x.shape[-1]
    Q, K, V = x @ Wq, x @ Wk, x @ Wv
    scores = (Q @ K.T) / np.sqrt(d)
    weights = softmax(scores)
    return weights @ V
