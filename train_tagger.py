# --- File: train_tagger.py (Cập nhật sử dụng Firebase/Firestore) ---
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
import firebase_admin
from firebase_admin import credentials, firestore
import os


# =======================================================================
# GIAI ĐOẠN 1: TẢI VÀ CHUẨN BỊ DỮ LIỆU TỪ FIREBASE
# =======================================================================
print(">>> Bắt đầu huấn luyện (training) AI Gắn Tag...")

# --- KẾT NỐI FIREBASE (Giữ nguyên phần bảo mật) ---
try:
    key_json_str = os.environ.get("FIREBASE_KEY_JSON")

    if not key_json_str:
        try:
            with open("serviceAccountKey.json", 'r') as f:
                key_json_dict = json.load(f)
        except FileNotFoundError:
            raise Exception("Lỗi: Không tìm thấy 'FIREBASE_KEY_JSON' trong biến môi trường và không tìm thấy file 'serviceAccountKey.json'.")
    else:
        key_json_dict = json.loads(key_json_str)

    cred = credentials.Certificate(key_json_dict)
    # Thay 'mealmaker-backend' bằng project ID thực tế của bạn
    firebase_admin.initialize_app(cred, {'projectId': 'mealmaker-backend'})
    db = firestore.client()
    print("Đã kết nối Firebase thành công.")

except Exception as e:
    print(f"Lỗi kết nối Firebase: {e}")
    sys.exit(1)
# ------------------------------------------------------------------------

# 1. Tải dữ liệu từ collection 'recipes'
recipes_data = []
try:
    print("Đang tải dữ liệu từ collection 'recipes'...")
    # ⭐️ ĐỌC DỮ LIỆU TỪ COLLECTION 'recipes'
    recipes_ref = db.collection("recipes").stream()

    for recipe_doc in recipes_ref:
        # doc.to_dict() trả về cấu trúc dữ liệu Map như bạn đã cung cấp
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

for recipe in recipes_data:
    # Gộp title, ingredients, instructions thành 1 khối văn bản
    title = recipe.get('title', '')
    # Sử dụng 'ingredients_list' hoặc 'ingredients_list_fixed'
    ingredients = " ".join(recipe.get('ingredients_list', recipe.get('ingredients_list_fixed', [])))
    instructions = " ".join(recipe.get('instructions', []))

    # Lấy time_minutes
    time_minutes = recipe.get('time_minutes', 0)

    # ----------------------------------------------------
    # 💡 LOGIC: TẠO VÀ GÁN TAG THỜI GIAN
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

    # Thêm tag thời gian vào danh sách tags, loại bỏ tags thời gian cũ
    tags = [t for t in tags if t not in ["quick", "medium_cook", "long_cook"]]
    if time_tag:
        tags.append(time_tag)

    # Gộp time_minutes vào input text để AI học tag thời gian
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
mlb = MultiLabelBinarizer()
y_binary = mlb.fit_transform(y_tags)
print(f'Đã "số hóa" nhãn (tags). Tìm thấy {len(mlb.classes_)} tags độc nhất.')

# 2. "Số hóa" Văn bản (Text)
vectorizer = TfidfVectorizer(max_features=5000, stop_words=None)
X_vectors = vectorizer.fit_transform(X_text)
print(f'Đã "số hóa" văn bản (TF-IDF).')


# =======================================================================
# GIAI ĐOẠN 3: HUẤN LUYỆN (TRAIN) MÔ HÌNH PHÂN LOẠI
# =======================================================================

# 1. Chọn mô hình
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