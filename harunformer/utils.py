import datetime
import re
import numpy as np

# 🧹 METİN TEMİZLEME
def clean_text(text: str) -> str:
    """
    Kullanıcı girdisini temizler:
    - Küçük harfe çevirir
    - Gereksiz karakterleri kaldırır
    """
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-ZçğıöşüÇĞİÖŞÜ\s]", "", text)
    return text


# 🧾 LOG KAYDETME
def save_log(user_input: str, bot_response: str, filename: str = "chat_log.txt"):
    """
    Kullanıcının ve botun konuşmalarını tarih damgası ile kaydeder.
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"[{now}] 👤 {user_input}\n[{now}] 🤖 {bot_response}\n\n")


# 📜 SON N KONUŞMAYI GÖRÜNTÜLEME
def show_logs(filename: str = "chat_log.txt", last_n: int = 5):
    """
    Kaydedilen konuşmalardan son N tanesini gösterir.
    """
    try:
        with open(filename, "r", encoding="utf-8") as f:
            lines = f.readlines()
            print("📜 Son konuşmalar:")
            print("".join(lines[-(last_n * 3):]))
    except FileNotFoundError:
        print("Henüz kayıtlı konuşma yok 😅")


# 📏 Vektör normalizasyonu (benzerlik hesaplamaları için)
def normalize(vec):
    """
    Vektörün uzunluğunu 1 yapar (unit vector).
    """
    norm = np.linalg.norm(vec) + 1e-8
    return vec / norm
