# --- File: app_ai_service.py (Python AI Service) ---
# SERVER PYTHON CHUYÊN BIỆT (VI DỊCH VỤ)

import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from datetime import datetime
import os          # <--- BỔ SUNG
import json        # <--- BỔ SUNG

# (MỚI) Import thư viện Firebase Admin
import firebase_admin
from firebase_admin import credentials, firestore

# =======================================================================
# KHỞI TẠO ỨNG DỤNG VÀ FIREBASE
# =======================================================================
app = Flask(__name__)
db = None  # Khởi tạo db mặc định là None

try:
    # 1. Đọc chuỗi JSON từ Biến Môi Trường
    key_json_str = os.environ.get("FIREBASE_KEY_JSON")

    if key_json_str:
        # Ưu tiên: Dùng Biến Môi Trường (từ môi trường deploy HOẶC .env)
        key_json_dict = json.loads(key_json_str)
        print(">>> Đang dùng khóa từ Biến Môi Trường/ .env <<<")
    else:
        # Fallback (Chỉ khi cần thiết): Tự đọc file serviceAccountKey.json
        # (Đây là backup nếu .env không hoạt động hoặc môi trường cần)
        try:
             with open("serviceAccountKey.json", 'r') as f:
                key_json_dict = json.load(f)
             print(">>> Đang dùng khóa từ file 'serviceAccountKey.json' (Fallback) <<<")
        except FileNotFoundError:
            raise Exception("Lỗi: Không tìm thấy khóa Firebase (Cả Biến Môi Trường và File cục bộ).")

    # 2. Chuyển chuỗi thành đối tượng Python Dictionary
    key_json_dict = json.loads(key_json_str)

    # 3. Khởi tạo chứng chỉ bằng nội dung Dict
    # Đây là cách AN TOÀN và CHÍNH XÁC để dùng Biến Môi Trường
    cred = credentials.Certificate(key_json_dict)

    # 4. Khởi tạo app (!! Dùng biến 'cred' vừa tạo)
    firebase_admin.initialize_app(cred, {
        'projectId': 'mealmaker-backend',
    })

    # 5. Lấy CSDL Firestore
    db = firestore.client()
    print(">>> KẾT NỐI FIREBASE (FIRESTORE) THÀNH CÔNG! <<<")

except Exception as e:
    # Log lỗi chi tiết, không chỉ cho người dùng mà còn cho việc debug
    print(f"LỖI: Không thể kết nối Firebase. Hãy kiểm tra biến môi trường 'FIREBASE_KEY_JSON'. Lỗi: {e}")
    # db đã được khởi tạo là None ở trên

# =======================================================================
# TẢI "BỘ NÃO" AI GỢI Ý (k-NN) VÀO BỘ NHỚ
# =======================================================================
model_prefix_knn = "ai_model_knn"
try:
    # (SỬA LẠI) Đổi tên biến (model -> model_knn)
    model_knn = joblib.load(f"{model_prefix_knn}_model.pkl")
    user_item_matrix = joblib.load(f"{model_prefix_knn}_matrix.pkl")
    recipe_id_map = joblib.load(f"{model_prefix_knn}_recipe_id_map.pkl")
    user_id_map = joblib.load(f"{model_prefix_knn}_user_id_map.pkl")
    print(f">>> TẢI 'BỘ NÃO AI k-NN' THÀNH CÔNG! (Đã huấn luyện {user_item_matrix.shape[0]} users) <<<")
except Exception as e:
    print(f"CẢNH BÁO: Không tải được 'bộ não' k-NN. (Hãy chạy 'train_model.py')")
    model_knn = None

# =======================================================================
# (MỚI) TẢI "BỘ NÃO" AI GẮN TAG (NLP) VÀO BỘ NHỚ
# =======================================================================
model_prefix_tagger = "ai_tagger"
try:
    model_tagger = joblib.load(f"{model_prefix_tagger}_model.pkl")
    vectorizer_tagger = joblib.load(f"{model_prefix_tagger}_vectorizer.pkl")
    mlb_tagger = joblib.load(f"{model_prefix_tagger}_mlb.pkl")
    print(f">>> TẢI 'BỘ NÃO AI GẮN TAG' (NLP) THÀNH CÔNG! <<<")
except Exception as e:
    print(f"CẢNH BÁO: Không tải được 'bộ não' Gắn Tag. (Hãy chạy 'train_tagger.py')")
    model_tagger = None

# =======================================================================
# (XÓA) AI 1: TỰ ĐỘNG GẮN TAG (DỰA TRÊN TỪ ĐIỂN)
# Chúng ta không cần TAG_DICTIONARY và hàm ai_auto_tag_recipe cũ nữa
# =======================================================================


# =======================================================================
# API ENDPOINTS (ĐỂ SERVER NODE.JS GỌI)
# =======================================================================

# API 1: (NÂNG CẤP) Dùng AI Huấn luyện để Gắn Tag
# Server Node.js sẽ gọi API này khi user upload bài
@app.route("/get-auto-tags", methods=["POST"])
def get_auto_tags():
    # Kiểm tra xem "bộ não" Gắn Tag đã được tải chưa
    if not model_tagger:
        return jsonify({"error": "Bộ não AI Gắn Tag (NLP) chưa sẵn sàng. Hãy chạy 'train_tagger.py'."}), 500

    try:
        data = request.json
        title = data.get("title", "")
        ingredients_list = data.get("ingredients_list", [])
        instructions_list = data.get("instructions", [])

        # (MỚI) Lấy time_minutes
        time = data.get("time_minutes", 0)

        # 1. Gộp text (giống hệt lúc train)
        ingredients_str = " ".join(ingredients_list)
        instructions_str = " ".join(instructions_list)

        # (ĐÃ SỬA) Gộp cả time_minutes vào input text để khớp với dữ liệu train
        full_text = f"{title} {ingredients_str} {instructions_str} time_{time}"

        # 2. Dùng "bộ não" (vectorizer) để biến text mới thành vector
        text_vector = vectorizer_tagger.transform([full_text])

        # 3. Dùng "bộ não" (model) để dự đoán vector nhãn
        # Dùng predict_proba và áp dụng ngưỡng
        predicted_probabilities = model_tagger.predict_proba(text_vector)

        # Áp dụng Ngưỡng Xác suất: Điều chỉnh ngưỡng này (ví dụ: 0.25 đến 0.4)
        THRESHOLD = 0.3
        predicted_binary = (predicted_probabilities > THRESHOLD).astype(int)

        # 4. Dùng "bộ não" (mlb) để dịch ngược vector nhãn thành chữ
        predicted_tags = mlb_tagger.inverse_transform(predicted_binary)

        # 5. Khởi tạo danh sách tags cuối cùng
        final_tags = list(predicted_tags[0])

        # =======================================================================
        # 💡 LOGIC KHẮC PHỤC LỖI TAG THỜI GIAN (HARD RULE)
        # =======================================================================

        TIME_TAGS = ["quick", "medium_cook", "long_cook"]

        # A. Loại bỏ bất kỳ tag thời gian nào AI VỪA DỰ ĐOÁN (để tránh tag sai)
        final_tags = [tag for tag in final_tags if tag not in TIME_TAGS]

        # B. Áp dụng luật cứng để TÍNH TOÁN tag thời gian chính xác
        time_tag_rule = None
        if time > 0 and time <= 25:
            time_tag_rule = "quick"         # Nhanh: <= 25 phút
        elif time > 25 and time <= 60:
            time_tag_rule = "medium_cook"   # Vừa: > 25 phút và <= 60 phút
        elif time > 60:
            time_tag_rule = "long_cook"     # Lâu: > 60 phút

        # C. Thêm tag thời gian chính xác vào danh sách cuối cùng
        if time_tag_rule:
            final_tags.append(time_tag_rule)

        print(f"AI (NLP) đã dự đoán các tag: {final_tags}")
        return jsonify(final_tags), 200

    except Exception as e:
        print(f"Lỗi khi gắn tag (NLP): {e}")
        return jsonify({"error": str(e)}), 500

# API 2: (SỬA LẠI) Dùng để Gợi ý (Hybrid AI)
# Server Node.js sẽ gọi API này
@app.route("/get-recommendations-ai", methods=["GET"])
def get_recommendations_ai():
    if not db: return jsonify({"error": "Lỗi CSDL"}), 500

    # (SỬA LẠI) Kiểm tra 'model_knn'
    if not model_knn: return jsonify({"error": "Bộ não AI k-NN chưa sẵn sàng."}), 500

    user_id = request.args.get("userId")
    if not user_id:
        return jsonify({"error": "Cần cung cấp userId"}), 400

    recommendation_ids = []

    # --- 1. THỬ DÙNG AI HỌC MÁY (k-NN) TRƯỚC (Cho user cũ) ---
    # Kiểm tra xem user_id có trong 'bộ não' không
    if user_id in user_id_map:
        try:
            print(f"Đang tìm gợi ý cho User Cũ ({user_id}) bằng k-NN...")
            user_index = user_id_map.get_loc(user_id)
            user_vector = user_item_matrix.iloc[user_index].values.reshape(1, -1)

            # (SỬA LẠI) Dùng 'model_knn'
            distances, indices = model_knn.kneighbors(user_vector, n_neighbors=15)

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
    # Nếu AI Học máy không tìm thấy gợi ý
    # HOẶC nếu user là người mới (không có trong 'user_id_map')
    if not recommendation_ids:
        print(f"k-NN thất bại (hoặc user mới). Chuyển sang AI Dựa trên Luật (Habits)...")
        try:
            user_doc = db.collection("users").document(user_id).get()

            if not user_doc.exists or "ai_profile" not in user_doc.to_dict():
                return jsonify({"error": "User mới, chưa chọn sở thích (Habits)"}), 404

            ai_profile = user_doc.to_dict()["ai_profile"]

            # Đọc đúng các trường 'diet' và 'favorite_cuisines'
            tags_to_query = [
                *(ai_profile.get("diet", [])),
                *(ai_profile.get("favorite_cuisines", []))
            ]

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
    return "Chào! Server AI (Python Hybrid Microservice v2) đang chạy!"

# =======================================================================
# CHẠY SERVER
# =======================================================================
if __name__ == "__main__":
    # (MỚI) Chạy trên cổng 5002
    app.run(debug=True, port=5002)