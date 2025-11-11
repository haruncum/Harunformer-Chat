import json
from harunformer.model import Harunformer

# Kelimeleri sayısal vektöre çevirecek basit bir sözlük
vocab = {"merhaba":0, "nasılsın":1, "iyiyim":2, "kötü":3, "selam":4}
model = Harunformer(vocab)

# Cevapları data klasöründen al
with open("data/responses.json", "r", encoding="utf-8") as f:
    responses = json.load(f)

print("🤖 Harunformer Chat'e hoş geldin! (çıkmak için 'q')")
while True:
    text = input("👤 Sen: ")
    if text.lower() == "q":
        break
    reply = model.generate_response(text, responses)
    print(" Harunformer:", reply)
