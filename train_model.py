# --- File: train_model.py (Python) ---
# KỊCH BẢN HUẤN LUYỆN (OFFLINE)
# (Đã cập nhật để khớp với Model của bạn)

import pandas as pd
from sklearn.neighbors import NearestNeighbors
import joblib
import firebase_admin
from firebase_admin import credentials, firestore

# =======================================================================
# GIAI ĐOẠN 1: TẢI DỮ LIỆU TỪ FIREBASE
# =======================================================================
print(">>> Bắt đầu huấn luyện (training)...")
try:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred, {'projectId': 'mealmaker-backend'}) # !! THAY THẾ
    db = firestore.client()
    print("Đã kết nối Firebase.")
except Exception as e:
    print(f"Lỗi kết nối Firebase: {e}")
    exit()

# Tải TẤT CẢ dữ liệu "Yêu thích" từ CSDL
raw_data = []
try:
    # (SỬA LẠI) Đọc từ collection "recipeLikes" (khớp với model RecipeLike)
    interactions_ref = db.collection("recipeLikes").stream()

    for interaction in interactions_ref:
        doc = interaction.to_dict()

        # (SỬA LẠI) Dùng 'user_id' và 'recipe_id' (khớp với model RecipeLike)
        if doc.get("user_id") and doc.get("recipe_id"):
            raw_data.append({
                "userId": doc.get("user_id"), # Standardize to 'userId' for the matrix
                "itemId": doc.get("recipe_id")  # Standardize to 'itemId' for the matrix
            })

    if len(raw_data) < 10: # Cần ít nhất 10 lượt "Tim" để train
        print("Lỗi: Không đủ dữ liệu 'favorite' (dưới 10). Dừng huấn luyện.")
        exit()

    print(f"Đã tải {len(raw_data)} lượt tương tác (Yêu thích) từ CSDL.")

except Exception as e:
    print(f"Lỗi khi tải dữ liệu: {e}")
    exit()

df = pd.DataFrame(raw_data)
df['rating'] = 1 # Mặc định 'Tim' là 1 điểm

# =======================================================================
# GIAI ĐOẠN 2: TẠO MA TRẬN (USER-ITEM MATRIX)
# =======================================================================
try:
    user_item_matrix = df.pivot_table(
        index='userId',
        columns='itemId',
        values='rating',
        fill_value=0
    )
except Exception as e:
    print(f"Lỗi khi tạo ma trận: {e}.")
    exit()

print(f"Đã tạo ma trận ({user_item_matrix.shape[0]} users x {user_item_matrix.shape[1]} items).")

# =======================================================================
# GIAI ĐOẠN 3: HUẤN LUYỆN (TRAIN) MÔ HÌNH k-NN
# =======================================================================
model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5)
model.fit(user_item_matrix)
print("Đã huấn luyện mô hình k-NN (Lọc Cộng tác) thành công.")

# =======================================================================
# GIAI ĐOẠN 4: LƯU "BỘ NÃO"
# =======================================================================
output_prefix = "ai_model_knn" # Tiền tố file
try:
    joblib.dump(model, f"{output_prefix}_model.pkl")
    joblib.dump(user_item_matrix, f"{output_prefix}_matrix.pkl")
    joblib.dump(user_item_matrix.columns, f"{output_prefix}_recipe_id_map.pkl")
    joblib.dump(user_item_matrix.index, f"{output_prefix}_user_id_map.pkl")

    print(f"\n>>> THÀNH CÔNG! <<<")
    print(f"Đã lưu 'Bộ não AI' thành các file {output_prefix}_*.pkl")
    print("Bây giờ bạn có thể chạy 'app_recommend.py' (Server Python).")

except Exception as e:
    print(f"Lỗi khi lưu mô hình: {e}")