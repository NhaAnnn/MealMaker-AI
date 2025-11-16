# --- File: train_tagger.py (Phiên bản Chính xác: Lọc dữ liệu & Tối ưu AI) ---

import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import sys
import firebase_admin
from firebase_admin import credentials, firestore
import os
import numpy as np


# =======================================================================
# GIAI ĐOẠN 1: TẢI VÀ CHUẨN BỊ DỮ LIỆU TỪ FIREBASE
# =======================================================================
print(">>> Bắt đầu huấn luyện (training) AI Gắn Tag Chính xác...")

# --- KẾT NỐI FIREBASE (Giữ nguyên) ---
try:
    key_json_str = os.environ.get("FIREBASE_KEY_JSON")
    if not key_json_str:
        try:
            with open("firebase_admin.json", 'r') as f:
                key_json_dict = json.load(f)
        except FileNotFoundError:
            raise Exception("Lỗi: Không tìm thấy khóa Firebase.")
    else:
        key_json_dict = json.loads(key_json_str)

    cred = credentials.Certificate(key_json_dict)
    firebase_admin.initialize_app(cred, {'projectId': 'mealplanner2-9b7dd'})
    db = firestore.client()
    print("Đã kết nối Firebase thành công.")
except Exception as e:
    print(f"Lỗi kết nối Firebase: {e}")
    sys.exit(1)
# ------------------------------------------------------------------------

# Tải dữ liệu từ collection 'recipes' (Giữ nguyên)
recipes_data = []
try:
    print("Đang tải dữ liệu từ collection 'recipes'...")
    recipes_ref = db.collection("recipes").stream()
    for recipe_doc in recipes_ref:
        recipes_data.append(recipe_doc.to_dict())
    if not recipes_data:
        raise ValueError("Lỗi: Không tìm thấy công thức nào trong CSDL.")
    print(f"Đã tải thành công {len(recipes_data)} công thức từ Firebase.")
except Exception as e:
    print(f"LỖI: Không thể tải dữ liệu từ Firestore: {e}")
    sys.exit(1)


# 2. Chuẩn bị dữ liệu (Input và Output)
X_text = [] # Input (Văn bản)
y_tags = [] # Output (Tags)

# Định nghĩa từ khóa thịt để lọc dữ liệu huấn luyện
MEAT_KEYWORDS = ["pork", "beef", "chicken", "lamb", "mutton", "ribs", "steak", "thịt", "sườn", "bì", "trứng", "egg", "cha", "chả", "bò", "gà", "heo", "cá", "hải sản", "seafood"]
TIME_TAGS = ["quick", "medium_cook", "long_cook"]

for recipe in recipes_data:
    title = recipe.get('title', '')
    ingredients = " ".join(recipe.get('ingredients_list', recipe.get('ingredients_list_fixed', [])))
    # instructions = " ".join(recipe.get('instructions', []))
    time_minutes = recipe.get('time_minutes', 0)

    tags = recipe.get('tags', [])

    # Loại bỏ tags thời gian cũ
    tags = [t for t in tags if t not in TIME_TAGS]

    # Tạo full_text
    full_text = f"{title} {ingredients}  time_{time_minutes}"

    # ----------------------------------------------------
    # ⭐️ CẢI TIẾN QUAN TRỌNG: LỌC TAG VEGETARIAN TRONG DỮ LIỆU HUẤN LUYỆN ⭐️
    # Đảm bảo tính nhất quán: Món có thịt KHÔNG THỂ là Vegetarian
    # ----------------------------------------------------
    is_meat_present = any(keyword in full_text.lower() for keyword in MEAT_KEYWORDS)

    if is_meat_present:
        # Nếu có thịt/trứng, loại bỏ tag 'vegetarian' và 'vegan' khỏi y_tags (Làm sạch dữ liệu)
        if "vegetarian" in tags: tags.remove("vegetarian")
        if "vegan" in tags: tags.remove("vegan")
    # ----------------------------------------------------

    # Gán tag thời gian (Dùng luật cứng cho quá trình huấn luyện)
    time_tag = None
    if time_minutes > 0 and time_minutes <= 25:
        time_tag = "quick"
    elif time_minutes > 25 and time_minutes <= 60:
        time_tag = "medium_cook"
    elif time_minutes > 60:
        time_tag = "long_cook"

    if time_tag:
        tags.append(time_tag)

    if full_text and tags:
        X_text.append(full_text)
        y_tags.append(tags)

print(f"Đã chuẩn bị {len(X_text)} mẫu dữ liệu hợp lệ để huấn luyện.")

# =======================================================================
# GIAI ĐOẠN 2: "SỐ HÓA" (VECTOR HÓA) DỮ LIỆU
# =======================================================================

# 1. "Số hóa" Nhãn (Tags)
mlb = MultiLabelBinarizer()
y_binary = mlb.fit_transform(y_tags)
print(f'Đã "số hóa" nhãn (tags). Tìm thấy {len(mlb.classes_)} tags độc nhất.')

# 2. "Số hóa" Văn bản (Text)
# ⭐️ SỬ DỤNG N-GRAMS (1, 2) ⭐️
vectorizer = TfidfVectorizer(max_features=5000,
                             stop_words=None,
                             ngram_range=(1, 2)) # Thêm Bigrams
X_vectors = vectorizer.fit_transform(X_text)
print(f'Đã "số hóa" văn bản (TF-IDF với N-grams).')


# =======================================================================
# GIAI ĐOẠN 3: HUẤN LUYỆN (TRAIN) MÔ HÌNH PHÂN LOẠI
# =======================================================================

# ⭐️ SỬ DỤNG TRỌNG SỐ LỚP CÂN BẰNG ⭐️
classifier = OneVsRestClassifier(LogisticRegression(solver='liblinear',
                                                    class_weight='balanced', # Cân bằng trọng số cho các tag hiếm
                                                    C=1.0,
                                                    max_iter=500),
                                 n_jobs=-1)

print("Bắt đầu huấn luyện mô hình phân loại với Trọng số Lớp Cân bằng...")
try:
    classifier.fit(X_vectors, y_binary)
    print("Đã huấn luyện mô hình phân loại thành công.")
except Exception as e:
    print(f"LỖI trong quá trình huấn luyện: {e}")
    sys.exit(1)


# =======================================================================
# GIAI ĐOẠN 4: LƯU "BỘ NÃO" GẮN TAG
# =======================================================================
output_prefix = "ai_tagger"
try:
    joblib.dump(classifier, f"{output_prefix}_model.pkl")
    joblib.dump(vectorizer, f"{output_prefix}_vectorizer.pkl")
    joblib.dump(mlb, f"{output_prefix}_mlb.pkl")

    print(f"\n>>> THÀNH CÔNG! <<<")
    print(f"Đã lưu 'Bộ não AI Gắn Tag' thành các file {output_prefix}_*.pkl")

except Exception as e:
    print(f"Lỗi khi lưu mô hình: {e}")