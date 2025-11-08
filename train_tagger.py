# --- File: train_tagger.py ---
# KỊCH BẢN HUẤN LUYỆN (OFFLINE) CHO AI GẮN TAG
# Chạy file này 1 lần (python train_tagger.py)

import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
import joblib

# =======================================================================
# GIAI ĐOẠN 1: TẢI VÀ CHUẨN BỊ DỮ LIỆU
# =======================================================================
print(">>> Bắt đầu huấn luyện (training) AI Gắn Tag...")

try:
    # 1. Tải dữ liệu từ recipes.json
    with open('recipes.json', 'r', encoding='utf-8') as f:
        recipes_data = json.load(f)
    print(f"Đã tải {len(recipes_data)} công thức từ 'recipes.json'.")
except FileNotFoundError:
    print("LỖI: Không tìm thấy file 'recipes.json'.")
    print("Hãy sao chép 'recipes.json' vào thư mục này.")
    exit()
except Exception as e:
    print(f"Lỗi khi đọc file JSON: {e}")
    exit()

# 2. Chuẩn bị dữ liệu (Input và Output)
X_text = [] # Input (Văn bản)
y_tags = [] # Output (Tags)

for recipe in recipes_data:
    # Gộp title, ingredients, instructions thành 1 khối văn bản
    title = recipe.get('title', '')
    ingredients = " ".join(recipe.get('ingredients_list', []))
    instructions = " ".join(recipe.get('instructions', []))

    full_text = f"{title} {ingredients} {instructions}"

    tags = recipe.get('tags', [])

    # Chỉ train những món có cả text và tags
    if full_text and tags:
        X_text.append(full_text)
        y_tags.append(tags)

print(f"Đã chuẩn bị {len(X_text)} mẫu dữ liệu hợp lệ để huấn luyện.")

# =======================================================================
# GIAI ĐOẠN 2: "SỐ HÓA" (VECTOR HÓA) DỮ LIỆU
# =======================================================================

# 1. "Số hóa" Nhãn (Tags)
# Dạy cho AI biết tất cả các tags có thể có
mlb = MultiLabelBinarizer()
y_binary = mlb.fit_transform(y_tags)
print(f'Đã "số hóa" nhãn (tags). Tìm thấy {len(mlb.classes_)} tags độc nhất.')
# y_binary bây giờ là [[1, 0, 1], [0, 1, 1], ...]

# 2. "Số hóa" Văn bản (Text)
# Dạy cho AI "đọc" văn bản
vectorizer = TfidfVectorizer(max_features=5000, stop_words=None) # Lấy 5000 từ quan trọng nhất
X_vectors = vectorizer.fit_transform(X_text)
print(f'Đã "số hóa" văn bản (TF-IDF).')

# =======================================================================
# GIAI ĐOẠN 3: HUẤN LUYỆN (TRAIN) MÔ HÌNH PHÂN LOẠI
# =======================================================================

# 1. Chọn mô hình
# OneVsRestClassifier: Biến bài toán "đa nhãn" (chọn nhiều tag)
# thành nhiều bài toán "nhị phân" (có tag này hay không?)
# LogisticRegression: Một mô hình phân loại nhanh và hiệu quả
classifier = OneVsRestClassifier(LogisticRegression(solver='liblinear'), n_jobs=-1)

# 2. Huấn luyện (Train)
print("Bắt đầu huấn luyện (train) mô hình phân loại... (Việc này có thể mất vài phút)")
classifier.fit(X_vectors, y_binary)
print("Đã huấn luyện mô hình phân loại thành công.")

# =======================================================================
# GIAI ĐOẠN 4: LƯU "BỘ NÃO" GẮN TAG
# =======================================================================
output_prefix = "ai_tagger" # Tiền tố file
try:
    # 1. Lưu mô hình Phân loại (bộ não chính)
    joblib.dump(classifier, f"{output_prefix}_model.pkl")
    # 2. Lưu bộ "Vector hóa" (để biến text mới thành số)
    joblib.dump(vectorizer, f"{output_prefix}_vectorizer.pkl")
    # 3. Lưu bộ "Nhị phân hóa" (để biến số dự đoán thành chữ)
    joblib.dump(mlb, f"{output_prefix}_mlb.pkl")

    print(f"\n>>> THÀNH CÔNG! <<<")
    print(f"Đã lưu 'Bộ não AI Gắn Tag' thành các file {output_prefix}_*.pkl")
    print("Bây giờ bạn có thể chạy 'app_ai_service.py' (Server Python).")

except Exception as e:
    print(f"Lỗi khi lưu mô hình: {e}")