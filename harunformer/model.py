import numpy as np
from .layers import transformer_block

class Harunformer:
    def __init__(self, vocab, embed_dim=8):
        self.vocab = vocab
        self.embed_dim = embed_dim
        # Kelime embedding'leri (rastgele başlangıç)
        self.embed = np.random.randn(len(vocab), embed_dim) * 0.1

    def encode(self, text):
        # Metni token'lara çevir
        tokens = [self.vocab.get(word, 0) for word in text.lower().split()]
        if not tokens:
            tokens = [0]
        # (seq_len, embed_dim)
        return np.array([self.embed[t] for t in tokens])

    def sentence_embedding(self, text):
        """
        Metni Transformer bloğundan geçirip
        tek bir vektöre indiriyoruz (ortalama pooling).
        """
        x = self.encode(text)             # (seq_len, embed_dim)
        if x.ndim == 1:
            x = x[None, :]
        x = transformer_block(x)          # (seq_len, embed_dim)
        return x.mean(axis=0)             # (embed_dim,)

    def similarity(self, text, key_text):
        """
        İki metin arasındaki kosinüs benzerliği.
        """
        v1 = self.sentence_embedding(text)
        v2 = self.sentence_embedding(key_text)
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        return float(np.dot(v1, v2) / denom)

    def generate_response(self, text, responses):
        """
        Kullanıcının mesajını, responses.json içindeki
        anahtar kelimelerle (merhaba, selam, hava, mutlu, üzgün, ...) 
        karşılaştırır. En benzer kategoriye ait cevabı döndürür.
        """
        best_key = "default"
        best_score = -1e9

        for key in responses.keys():
            score = self.similarity(text, key)
            if score > best_score:
                best_score = score
                best_key = key

        candidates = responses.get(best_key, responses["default"])
        return np.random.choice(candidates)
