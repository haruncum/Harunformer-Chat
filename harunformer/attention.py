import numpy as np

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

def self_attention(x):
    """
    x: (seq_len, embed_dim)
    Basit self-attention: Q = K = V = x
    """
    d = x.shape[-1]

    Q = x
    K = x
    V = x

    # Attention skorları
    scores = Q @ K.T / np.sqrt(d)  # (seq_len, seq_len)

    # Ağırlıklar (softmax)
    weights = softmax(scores)      # (seq_len, seq_len)

    # Çıktı = ağırlıklı toplam
    out = weights @ V              # (seq_len, embed_dim)

    return out, weights
