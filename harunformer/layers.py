import numpy as np
from .attention import self_attention

def layer_norm(x, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + eps)

def feed_forward(x, W1, W2):
    return np.maximum(0, x @ W1) @ W2

def transformer_block(x, params):
    Wq, Wk, Wv, W1, W2 = params
    attn_out = self_attention(x, Wq, Wk, Wv)
    x = layer_norm(x + attn_out)
    ff_out = feed_forward(x, W1, W2)
    return layer_norm(x + ff_out)
