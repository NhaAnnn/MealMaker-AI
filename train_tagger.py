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
import sys


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
    print("Hãy đảm bảo file 'recipes.json' chứa dữ liệu đầy đủ.")
    sys.exit(1)
except Exception as e:
    print(f"Lỗi khi đọc file JSON: {e}")
    sys.exit(1)

# 2. Chuẩn bị dữ liệu (Input và Output)
X_text = [] # Input (Văn bản)
y_tags = [] # Output (Tags)

for recipe in recipes_data:
    # Gộp title, ingredients, instructions thành 1 khối văn bản
    title = recipe.get('title', '')
    ingredients = " ".join(recipe.get('ingredients_list', recipe.get('ingredients_list_fixed', [])))
    instructions = " ".join(recipe.get('instructions', []))

    # Lấy time_minutes
    time_minutes = recipe.get('time_minutes', 0)

    # ----------------------------------------------------
    # 💡 LOGIC MỚI: TẠO VÀ GÁN TAG THỜI GIAN
    # ----------------------------------------------------

    # Lấy các tags hiện có
    tags = recipe.get('tags', [])

    # Tạo tag thời gian mới
    time_tag = None
    if time_minutes > 0 and time_minutes <= 25:
        time_tag = "quick"         # Nhanh: <= 25 phút
    elif time_minutes > 25 and time_minutes <= 60:
        time_tag = "medium_cook"   # Vừa: > 25 phút và <= 60 phút
    elif time_minutes > 60:
        time_tag = "long_cook"     # Lâu: > 60 phút

    # Thêm tag thời gian vào danh sách tags
    if time_tag and time_tag not in tags:
        tags.append(time_tag)

    # CŨNG CÓ THỂ BỎ TAG "quick" ĐÃ CÓ VÀ THAY THẾ BẰNG TAG PHÙ HỢP
    # Nếu muốn đảm bảo chỉ có một tag thời gian:
    # tags = [t for t in tags if t not in ["quick", "medium_cook", "long_cook"]]
    # if time_tag: tags.append(time_tag)

    # Gộp time_minutes vào input text để AI học tag thời gian
    # Việc này vẫn cần thiết để AI có thể nhìn thấy giá trị số
    full_text = f"{title} {ingredients} {instructions} time_{time_minutes}"

    # ----------------------------------------------------

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

# 2. "Số hóa" Văn bản (Text)
# Dạy cho AI "đọc" văn bản
# Giữ nguyên max_features=5000
vectorizer = TfidfVectorizer(max_features=5000, stop_words=None)
X_vectors = vectorizer.fit_transform(X_text)
print(f'Đã "số hóa" văn bản (TF-IDF).')


# =======================================================================
# GIAI ĐOẠN 3: HUẤN LUYỆN (TRAIN) MÔ HÌNH PHÂN LOẠI
# =======================================================================

# 1. Chọn mô hình
# OneVsRestClassifier cho phép sử dụng mô hình nhị phân (LogisticRegression)
# cho bài toán gắn nhãn đa nhãn.
classifier = OneVsRestClassifier(LogisticRegression(solver='liblinear'), n_jobs=-1)

# 2. Huấn luyện (Train)
print("Bắt đầu huấn luyện (train) mô hình phân loại... (Việc này có thể mất vài phút)")
try:
    classifier.fit(X_vectors, y_binary)
    print("Đã huấn luyện mô hình phân loại thành công.")
except Exception as e:
    print(f"LỖI trong quá trình huấn luyện: {e}")
    sys.exit(1)


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
    print("Bây giờ bạn có thể sử dụng các file này để dự đoán tag.")

except Exception as e:
    print(f"Lỗi khi lưu mô hình: {e}")