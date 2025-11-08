# --- File: app_ai_service.py (Python AI Service) ---
# SERVER PYTHON CHUYÊN BIỆT (VI DỊCH VỤ)
# (Phiên bản Nâng cấp: Tự xử lý User Cũ (ML) và User Mới (Rules))

import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from datetime import datetime

# (MỚI) Import thư viện Firebase Admin
import firebase_admin
from firebase_admin import credentials, firestore

# =======================================================================
# KHỞI TẠO ỨNG DỤNG VÀ FIREBASE
# =======================================================================
app = Flask(__name__)
try:
    # 1. Tải "chìa khóa" bí mật từ file JSON
    cred = credentials.Certificate("serviceAccountKey.json")

    # 2. Khởi tạo app (!! THAY THẾ BẰNG PROJECT ID CỦA BẠN)
    firebase_admin.initialize_app(cred, {
        'projectId': 'mealmaker-backend',
    })

    # 3. Lấy CSDL Firestore
    db = firestore.client()
    print(">>> KẾT NỐI FIREBASE (FIRESTORE) THÀNH CÔNG! <<<")

except Exception as e:
    print(f"LỖI: Không thể kết nối Firebase. Hãy kiểm tra file 'serviceAccountKey.json' và 'YOUR_PROJECT_ID'. Lỗi: {e}")
    db = None

# =======================================================================
# TẢI "BỘ NÃO" AI (ĐÃ HUẤN LUYỆN) VÀO BỘ NHỚ
# =======================================================================
model_prefix = "ai_model_knn"
try:
    model = joblib.load(f"{model_prefix}_model.pkl")
    user_item_matrix = joblib.load(f"{model_prefix}_matrix.pkl")
    recipe_id_map = joblib.load(f"{model_prefix}_recipe_id_map.pkl")
    user_id_map = joblib.load(f"{model_prefix}_user_id_map.pkl")
    print(f">>> TẢI 'BỘ NÃO AI' (k-NN) THÀNH CÔNG! (Đã huấn luyện {user_item_matrix.shape[0]} users) <<<")
except Exception as e:
    print(f"CẢNH BÁO: Không tải được file 'bộ não' {model_prefix}_*.pkl.")
    print(">>> AI Gợi ý (k-NN) sẽ không hoạt động. Hãy chạy file 'train_model.py' <<<")
    model = None

# =======================================================================
# AI 1: TỰ ĐỘNG GẮN TAG (ĐÃ "TRAIN" ĐỂ KHỚP VỚI MÀN HÌNH HABITS)
# =======================================================================
TAG_DICTIONARY = {
    "Gà": ["gà", "chicken", "cánh gà", "đùi gà", "ức gà"],
    "Bò": ["bò", "beef", "thăn bò", "bắp bò"],
    "Heo": ["heo", "lợn", "pork", "thịt ba chỉ", "sườn non"],
    "Cá": ["cá", "fish", "cá hồi", "cá basa", "cá diêu hồng"],
    "Hải Sản": ["tôm", "mực", "nghêu", "sò", "hải sản", "cua", "ghẹ"],
    "Nướng": ["nướng", "quay", "grill", "đút lò"],
    "Chiên": ["chiên", "rán", "fry"],
    "Xào": ["xào", "stir-fry"],
    "Hấp": ["hấp", "steam"],
    "Luộc": ["luộc", "boil"],
    "Kho": ["kho", "rim"],
    "Salad": ["salad", "gỏi", "trộn"],
    "Canh": ["canh", "soup", "súp"],
    "low_carb": ["low carb", "ít tinh bột", "keto", "giảm cân"],
    "high_protein": ["high protein", "nhiều đạm", "tăng cơ", "protein"],
    "vegetarian": ["chay", "vegetarian", "đậu phụ", "nấm", "rau củ", "đậu hũ"],
    "low_fat": ["low fat", "ít béo", "giảm mỡ"],
    "easy_level": ["dễ", "sơ cấp", "easy"],
    "advanced_level": ["nâng cao", "khó", "advanced", "cầu kỳ"],
    "vietnamese": ["việt nam", "vietnamese", "Việt", "phở", "bún"],
    "european": ["european", "âu", "mỹ", "ý", "pizza", "spaghetti", "bít tết"],
    "asian": ["asian", "nhật", "hàn", "trung", "thái", "kim chi", "sushi"],
    "latin": ["latin", "mexico", "tây ban nha", "taco"],
}

def ai_auto_tag_recipe(title, ingredients_str, time):
    tags = set()
    text_to_analyze = (title + " " + ingredients_str).lower()

    for tag, keywords in TAG_DICTIONARY.items():
        for keyword in keywords:
            if keyword in text_to_analyze:
                tags.add(tag)

    if any(t in tags for t in ["Gà", "Bò", "Heo"]):
        tags.add("Thịt")

    if time <= 15:
        tags.add("under_15")
    if time <= 30:
        tags.add("under_30")
    if time >= 60:
        tags.add("advanced_level")

    return list(tags)

# =======================================================================
# API ENDPOINTS (ĐỂ SERVER NODE.JS GỌI)
# =======================================================================

# API 1: (MỚI) Dùng để Gắn Tag
# Server Node.js sẽ gọi API này khi user upload bài
@app.route("/get-auto-tags", methods=["POST"])
def get_auto_tags():
    try:
        data = request.json
        title = data.get("title", "")
        # Nhận 'ingredients_list' (mảng) từ Node.js và chuyển thành chuỗi (string)
        ingredients_list = data.get("ingredients_list", [])
        ingredients_str = " ".join(ingredients_list)

        time = data.get("time_minutes", 0)

        # 1. Gọi AI Gắn Tag
        tags = ai_auto_tag_recipe(title, ingredients_str, time)

        # 2. Trả về danh sách tag
        return jsonify(tags), 200

    except Exception as e:
        print(f"Lỗi khi gắn tag: {e}")
        return jsonify({"error": str(e)}), 500


# API 2: (SỬA LẠI) Dùng để Gợi ý (Hybrid AI)
# Server Node.js sẽ gọi API này
@app.route("/get-recommendations-ai", methods=["GET"])
def get_recommendations_ai():
    if not db: return jsonify({"error": "Lỗi CSDL"}), 500

    user_id = request.args.get("userId")
    if not user_id:
        return jsonify({"error": "Cần cung cấp userId"}), 400

    recommendation_ids = []

    # --- 1. THỬ DÙNG AI HỌC MÁY (k-NN) TRƯỚC (Cho user cũ) ---
    # Kiểm tra xem 'model' đã được tải VÀ user_id có trong 'bộ não' không
    if model and user_id in user_id_map:
        try:
            print(f"Đang tìm gợi ý cho User Cũ ({user_id}) bằng k-NN...")
            user_index = user_id_map.get_loc(user_id)
            user_vector = user_item_matrix.iloc[user_index].values.reshape(1, -1)

            distances, indices = model.kneighbors(user_vector, n_neighbors=15) # Tăng "hàng xóm"

            recommendations = {}
            similar_user_indices = indices.flatten()[1:] # Bỏ qua user 0 (chính mình)

            for neighbor_index in similar_user_indices:
                neighbor_vector = user_item_matrix.iloc[neighbor_index]
                for recipe_index, rating in enumerate(neighbor_vector):
                    if rating == 1 and user_vector[0][recipe_index] == 0:
                        recipe_id = recipe_id_map[recipe_index]
                        recommendations[recipe_id] = recommendations.get(recipe_id, 0) + 1

            # Lấy 14 món (2 món/ngày x 7 ngày)
            sorted_ids = [r[0] for r in sorted(recommendations.items(), key=lambda item: item[1], reverse=True)[:14]]

            if sorted_ids:
                recommendation_ids = sorted_ids
        except Exception as e:
             print(f"Lỗi k-NN, chuyển sang fallback: {e}")
             recommendation_ids = [] # Đặt lại

    # --- 2. FALLBACK (DỰ PHÒNG) CHO USER MỚI (Dùng ai_profile) ---
    # Nếu AI Học máy không tìm thấy gợi ý (ví dụ: user cũ nhưng ít 'Tim')
    # HOẶC nếu user là người mới (không có trong 'user_id_map')
    if not recommendation_ids:
        print(f"k-NN thất bại (hoặc user mới). Chuyển sang AI Dựa trên Luật (Habits)...")
      try:
            user_doc = db.collection("users").document(user_id).get()

            if not user_doc.exists or "ai_profile" not in user_doc.to_dict():
                return jsonify({"error": "User mới, chưa chọn sở thích (Habits)"}), 404

            # --- (SỬA LẠI LOGIC TẠI ĐÂY) ---
            # Lấy tag từ cấu trúc ai_profile (Model User của bạn)
            ai_profile = user_doc.to_dict()["ai_profile"]

            # (SỬA LẠI) Đọc đúng các trường 'diet' và 'favorite_cuisines'
            tags_to_query = [
                *(ai_profile.get("diet", [])),
                *(ai_profile.get("favorite_cuisines", []))
                # (Chúng ta cũng có thể thêm "Tốc độ" (easy_level) nếu server Node.js
                # lưu nó vào ai_profile.cooking_skill_level)
            ]
            # --- KẾT THÚC SỬA LỖI ---

            if not tags_to_query:
                return jsonify({"error": "User chưa chọn sở thích nào"}), 404

            print(f"Đang tìm món ăn dựa trên sở thích: {tags_to_query}")

            # Query CSDL 'recipes'
            recommendations_ref = db.collection("recipes").where(
                "tags", "array-contains-any", tags_to_query
            ).limit(14).stream() # Lấy 14 món

            for r in recommendations_ref:
                recommendation_ids.append(r.to_dict().get("recipe_id")) # Lấy recipe_id

        except Exception as e:
             print(f"Lỗi khi dùng Fallback (Habits): {e}")
             return jsonify({"error": str(e)}), 500

    # --- 3. BƯỚC CUỐI: TRẢ VỀ DANH SÁCH ID ---
    if not recommendation_ids:
        return jsonify({"error": "Không tìm thấy gợi ý nào"}), 404

    # Trả về danh sách ID (Node.js sẽ tra cứu thông tin đầy đủ)
    return jsonify(recommendation_ids)


# API Ping
@app.route("/")
def hello():
    return "Chào! Server AI (Python Hybrid Microservice) đang chạy!"

# =======================================================================
# CHẠY SERVER
# =======================================================================
if __name__ == "__main__":
    # (MỚI) Chạy trên cổng 5002
    app.run(debug=True, port=5002)