import re
import string
from pathlib import Path
from underthesea import word_tokenize

# Lấy đường dẫn tuyệt đối của thư mục hiện tại (utils/)
BASE_DIR = Path(__file__).resolve().parent

# Trỏ đến thư mục files nằm cùng cấp với utils/
FILES_DIR = BASE_DIR.parent / "files"

# Tạo đường dẫn tuyệt đối
stop_word_file = FILES_DIR / "vietnamese-stopwords.txt"
emojicon_file = FILES_DIR / "emojicon.txt"
teencode_file = FILES_DIR / "teencode.txt"

# -----------------------
# 1. TẢI FILE NGUỒN
# -----------------------
with open(stop_word_file, 'r', encoding='utf-8') as f:
    stopwords = set([w.strip() for w in f.readlines() if w.strip()])

with open(emojicon_file, 'r', encoding='utf-8') as f:
    emojicons = [w.strip() for w in f.readlines() if w.strip()]

with open(teencode_file, 'r', encoding='utf-8') as f:
    teencode_map = {}
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            teencode_map[parts[0]] = " ".join(parts[1:])


special_tokens = ['', ' ', ',', '.', '...', '-', ':', ';', '?', '%', '(', ')', '+', '/', "'", '&', '#', '*', '!', '"', '_', '=', '[', ']', '{', '}', '~', '`', '|', '\\']


def remove_emojis(text):
    for emo in emojicons:
        text = text.replace(emo, ' ')
    return text

def normalize_teencode(text):
    for key, val in teencode_map.items():
        text = re.sub(rf'\b{re.escape(key)}\b', val, text)
    return text

def remove_special_chars(text):
    text = re.sub(r'[^\w\s]', ' ', text)  # loại ký tự đặc biệt
    text = re.sub(r'\s+', ' ', text).strip()  # loại khoảng trắng thừa
    return text

# -----------------------
# 4. TÁCH STOPWORD RIÊNG
# -----------------------
def remove_stopwords(text):
    tokens = word_tokenize(text, format="text").split()
    tokens = [t for t in tokens if t not in stopwords]
    return ' '.join(tokens)

# -----------------------
# 5. CHUẨN HÓA TỔNG HỢP
# -----------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = remove_emojis(text)
    text = normalize_teencode(text)
    text = remove_special_chars(text)
    text = remove_stopwords(text)
    return text