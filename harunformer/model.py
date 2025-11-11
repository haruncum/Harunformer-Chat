import numpy as np
from .layers import transformer_block

class Harunformer:
    def __init__(self, vocab, embed_dim=8):
        self.vocab = vocab
        self.embed_dim = embed_dim
        self.embed = np.random.randn(len(vocab), embed_dim) * 0.1
        self.params = [np.random.randn(embed_dim, embed_dim) * 0.1 for _ in range(5)]

    def encode(self, text):
        tokens = [self.vocab.get(word, 0) for word in text.lower().split()]
        return np.array([self.embed[t] for t in tokens])

    def detect_emotion(self, text):
        text_lower = text.lower()
        if any(word in text_lower for word in ["mutlu", "harika", "iyi", "güzel"]):
            return "mutlu"
        elif any(word in text_lower for word in ["üzgün", "kötü", "moral", "ağlıyorum", "sıkıldım"]):
            return "üzgün"
        elif any(word in text_lower for word in ["sinir", "öfke", "delirdim", "lanet"]):
            return "sinirli"
        else:
            return None

    def generate_response(self, text, responses):
        text_lower = text.lower()
        emotion = self.detect_emotion(text_lower)

        if emotion:
            return np.random.choice(responses[emotion])

        if "merhaba" in text_lower:
            return np.random.choice(responses["merhaba"])
        elif "selam" in text_lower:
            return np.random.choice(responses["selam"])
        elif "nasılsın" in text_lower:
            return np.random.choice(responses["nasılsın"])
        elif "naber" in text_lower:
            return np.random.choice(responses["naber"])
        elif "hava" in text_lower or "yağmur" in text_lower or "güneş" in text_lower:
            return np.random.choice(responses["hava"])
        else:
            return np.random.choice(responses["default"])
