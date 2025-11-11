import numpy as np
from .attention import self_attention

def layer_norm(x, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + eps)

def transformer_block(x):
    """
    Basitleştirilmiş bir Transformer encoder bloğu:
      - Self-Attention
      - Residual connection
      - Layer Normalization
    """
    attn_out, _ = self_attention(x)
    x = layer_norm(x + attn_out)
    return x
