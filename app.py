# --- File: app_ai_service.py (Python AI Service) ---
# SERVER PYTHON CHUYÊN BIỆT (VI DỊCH VỤ)

import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from datetime import datetime
import os
import json
import random
# ⭐ THÊM FIELD FILTER để khắc phục UserWarning Firestore
from google.cloud.firestore import FieldFilter

# Import thư viện Firebase Admin
import firebase_admin
from firebase_admin import credentials, firestore
from firebase_admin import db as firebase_rtdb # Thêm alias nếu cần dùng Realtime DB

# =======================================================================
# KHỞI TẠO ỨNG DỤNG VÀ FIREBASE
# =======================================================================
app = Flask(__name__)
db = None  # Khởi tạo db mặc định là None

# Tên file chứng chỉ Firebase (chỉ dùng khi phát triển cục bộ)
FIREBASE_LOCAL_KEY_FILE = "firebase_admin.json"
FIREBASE_PROJECT_ID = 'mealplanner2-9b7dd'

try:
    # 1. Đọc chuỗi JSON từ Biến Môi Trường hoặc file cục bộ
    key_json_str = os.environ.get("FIREBASE_KEY_JSON")
    key_json_dict = None

    if key_json_str:
        # Ưu tiên: Dùng Biến Môi Trường (từ môi trường deploy HOẶC .env)
        key_json_dict = json.loads(key_json_str)
        print(">>> Đang dùng khóa từ Biến Môi Trường/ .env <<<")
    else:
        # Fallback: Tự đọc file serviceAccountKey.json
        try:
            with open(FIREBASE_LOCAL_KEY_FILE, 'r') as f:
                key_json_dict = json.load(f)
            print(f">>> Đang dùng khóa từ file '{FIREBASE_LOCAL_KEY_FILE}' (Fallback) <<<")
        except FileNotFoundError:
            raise Exception("Lỗi: Không tìm thấy khóa Firebase (Cả Biến Môi Trường và File cục bộ).")

    # 2. Khởi tạo chứng chỉ và app
    cred = credentials.Certificate(key_json_dict)

    # 3. Khởi tạo app (!! Dùng biến 'cred' vừa tạo)
    firebase_admin.initialize_app(cred, {
        'projectId': FIREBASE_PROJECT_ID,
    })

    # 4. Lấy CSDL Firestore
    db = firestore.client()
    print(">>> KẾT NỐI FIREBASE (FIRESTORE) THÀNH CÔNG! <<<")

except Exception as e:
    # Log lỗi chi tiết
    print(f"LỖI: Không thể kết nối Firebase. Hãy kiểm tra biến môi trường 'FIREBASE_KEY_JSON'. Lỗi: {e}")
    # db vẫn là None nếu lỗi

# =======================================================================
# TẢI "BỘ NÃO" AI GỢI Ý (k-NN) VÀO BỘ NHỚ
# =======================================================================
model_knn = None
user_item_matrix = None
recipe_id_map = None
user_id_map = None
model_prefix_knn = "ai_model_knn"
try:
    model_knn = joblib.load(f"{model_prefix_knn}_model.pkl")
    user_item_matrix = joblib.load(f"{model_prefix_knn}_matrix.pkl")
    user_id_map = joblib.load(f"{model_prefix_knn}_user_id_map.pkl")
    recipe_id_map = joblib.load(f"{model_prefix_knn}_recipe_id_map.pkl")
    print(f">>> TẢI 'BỘ NÃO AI k-NN' THÀNH CÔNG! (Đã huấn luyện {user_item_matrix.shape[0]} users) <<<")
except Exception as e:
    print(f"CẢNH BÁO: Không tải được 'bộ não' k-NN. (Lỗi: {e})")

# =======================================================================
# TẢI "BỘ NÃO" AI GẮN TAG (NLP) VÀO BỘ NHỚ
# =======================================================================
model_tagger = None
vectorizer_tagger = None # Khai báo để dùng global
mlb_tagger = None       # Khai báo để dùng global
model_prefix_tagger = "ai_tagger"
try:
    model_tagger = joblib.load(f"{model_prefix_tagger}_model.pkl")
    vectorizer_tagger = joblib.load(f"{model_prefix_tagger}_vectorizer.pkl")
    mlb_tagger = joblib.load(f"{model_prefix_tagger}_mlb.pkl")
    print(f">>> TẢI 'BỘ NÃO AI GẮN TAG' (NLP) THÀNH CÔNG! <<<")
except Exception as e:
    print(f"CẢNH BÁO: Không tải được 'bộ não' Gắn Tag. (Lỗi: {e})")

# =======================================================================
# API ENDPOINTS
# =======================================================================

# API 1: Dùng AI Huấn luyện để Gắn Tag
@app.route("/get-auto-tags", methods=["POST"])
def get_auto_tags():
    # ⭐ Đảm bảo mô hình NLP đã được tải
    if not model_tagger or not vectorizer_tagger or not mlb_tagger:
        return jsonify({"error": "Bộ não AI Gắn Tag (NLP) chưa sẵn sàng. Hãy kiểm tra file mô hình."}), 500

    try:
        data = request.json
        title = data.get("title", "")
        ingredients_list = data.get("ingredients_list", [])
        # instructions_list = data.get("instructions", []) # Hiện tại không dùng
        time = data.get("time_minutes", 0)

        # 1. Gộp text (giống hệt lúc train)
        ingredients_str = " ".join(ingredients_list)
        full_text = f"{title} {ingredients_str} time_{time}"

        # 2. Dùng "bộ não" (vectorizer) để biến text mới thành vector
        text_vector = vectorizer_tagger.transform([full_text])

        # 3. Dùng "bộ não" (model) để dự đoán vector nhãn
        predicted_probabilities = model_tagger.predict_proba(text_vector)
        THRESHOLD = 0.33
        predicted_binary = (predicted_probabilities > THRESHOLD).astype(int)

        # 4. Dùng "bộ não" (mlb) để dịch ngược vector nhãn thành chữ
        predicted_tags = mlb_tagger.inverse_transform(predicted_binary)
        final_tags = list(predicted_tags[0])

        # 5. LOGIC KHẮC PHỤC LỖI TAG THỜI GIAN (HARD RULE)
        TIME_TAGS = ["quick", "medium_cook", "long_cook"]
        final_tags = [tag for tag in final_tags if tag not in TIME_TAGS] # A. Loại bỏ tag AI dự đoán

        time_tag_rule = None
        if time > 0 and time <= 25:
            time_tag_rule = "quick"
        elif time > 25 and time <= 60:
            time_tag_rule = "medium_cook"
        elif time > 60:
            time_tag_rule = "long_cook"

        if time_tag_rule:
            final_tags.append(time_tag_rule) # C. Thêm tag thời gian chính xác

        print(f"AI (NLP) đã dự đoán các tag: {final_tags}")
        return jsonify(final_tags), 200

    except Exception as e:
        print(f"Lỗi khi gắn tag (NLP): {e}")
        return jsonify({"error": str(e)}), 500

# API 2: Dùng để Gợi ý (Hybrid AI)
@app.route("/get-recommendations-ai", methods=["GET"])
def get_recommendations_ai():
    # ⭐ Đảm bảo kết nối DB đã thành công
    if not db: return jsonify({"error": "Lỗi kết nối CSDL Firestore. Vui lòng kiểm tra khóa Firebase."}), 500

    user_id = request.args.get("userId")
    if not user_id:
        return jsonify({"error": "Cần cung cấp userId"}), 400

    recommendation_ids = []

    # --- 1. LEVEL 1: DÙNG AI HỌC MÁY (k-NN) ---
    # Kiểm tra xem mô hình k-NN đã load và user có trong ma trận hay không
    if model_knn and user_id in user_id_map.index: # ⭐ Dùng .index để kiểm tra
        try:
            N_NEIGHBORS = 15
            N_POTENTIAL = 50
            K_FINAL = 14

            print(f"Đang tìm gợi ý cho User Cũ ({user_id}) bằng k-NN...")
            user_index = user_id_map.get_loc(user_id)
            user_vector = user_item_matrix.iloc[user_index].values.reshape(1, -1)
            distances, indices = model_knn.kneighbors(user_vector, n_neighbors=N_NEIGHBORS)

            recommendations = {}
            similar_user_indices = indices.flatten()[1:]

            for neighbor_index in similar_user_indices:
                neighbor_vector = user_item_matrix.iloc[neighbor_index]
                for recipe_index, rating in enumerate(neighbor_vector):
                    # Chỉ lấy món mà người dùng hiện tại CHƯA đánh giá (rating == 0)
                    if rating == 1 and user_vector[0][recipe_index] == 0:
                        # Lấy ID của món ăn từ map
                        recipe_id = recipe_id_map[recipe_index]
                        recommendations[recipe_id] = recommendations.get(recipe_id, 0) + 1

            # Sắp xếp và lấy N_POTENTIAL món có tần suất xuất hiện cao nhất
            potential_ids = [r[0] for r in sorted(recommendations.items(),
                                                  key=lambda item: item[1],
                                                  reverse=True)[:N_POTENTIAL]]

            if potential_ids:
                random.shuffle(potential_ids)
                recommendation_ids = potential_ids[:K_FINAL]
                print(f"k-NN đã tìm thấy {len(recommendation_ids)} món (sau ngẫu nhiên hóa).")

            if not recommendation_ids:
                print("k-NN không tìm thấy đủ gợi ý. Chuyển sang fallback.")
        except Exception as e:
            print(f"Lỗi k-NN, chuyển sang fallback: {e}")
            recommendation_ids = []

    # --- 2. LEVEL 2: FALLBACK DỰA TRÊN LUẬT (HABITS/SỞ THÍCH) ---
    if not recommendation_ids:
        print(f"Level 2: Chuyển sang AI Dựa trên Luật (Habits)...")
        try:
            user_doc = db.collection("users").document(user_id).get()

            if not user_doc.exists:
                print("User chưa tồn tại trong DB hoặc chưa có ai_profile. Bỏ qua Level 2.")
                pass
            else:
                ai_profile = user_doc.to_dict().get("ai_profile", {})
                user_tags_array = ai_profile.get("tags", [])
                tags_to_query = [item.get("tag_name") for item in user_tags_array if item.get("tag_name")]

                if not tags_to_query:
                    print("User chưa chọn sở thích nào. Bỏ qua Level 2.")
                    pass
                else:
                    print(f"Đang tìm món ăn dựa trên sở thích: {tags_to_query}")

                    # ⭐ ĐÃ SỬA: Dùng cú pháp filter=FieldFilter để loại bỏ UserWarning
                    recommendations_ref = db.collection("recipes").where(
                        filter=FieldFilter("tags", "array_contains_any", tags_to_query)
                    ).limit(50).stream()

                    fallback_potential_ids = []
                    for r in recommendations_ref:
                        fallback_potential_ids.append(r.id) # Lấy Document ID bằng .id

                    if fallback_potential_ids:
                        random.shuffle(fallback_potential_ids)
                        recommendation_ids = fallback_potential_ids[:14]
                        print(f"Fallback đã chọn ngẫu nhiên {len(recommendation_ids)} món từ {len(fallback_potential_ids)} tiềm năng.")

        except Exception as e:
            print(f"Lỗi khi dùng Fallback (Habits): {e}")

    # --- 3. LEVEL 3: FALLBACK TOÀN CẦU/NGẪU NHIÊN (Chống lỗi 404) ---
    if not recommendation_ids:
        print("Level 3 Fallback: Đang lấy ngẫu nhiên 14 món phổ biến...")
        try:
            # Lấy 50 món (ví dụ: có 'views' cao nhất, hoặc mới nhất)
            global_recommendations_ref = db.collection("recipes").limit(50).stream()

            global_potential_ids = [r.id for r in global_recommendations_ref]

            if global_potential_ids:
                random.shuffle(global_potential_ids)
                # Chỉ lấy tối đa 14 món
                recommendation_ids = global_potential_ids[:14]
                print(f"Level 3 Fallback đã chọn ngẫu nhiên {len(recommendation_ids)} món từ danh sách toàn cầu.")

        except Exception as e:
            print(f"Lỗi Level 3 Fallback: {e}")

    # --- 4. BƯỚC CUỐI: TRẢ VỀ DANH SÁCH ID ---
    if not recommendation_ids:
        return jsonify({"error": "Không tìm thấy gợi ý nào"}), 404

    # Trả về danh sách ID
    return jsonify(recommendation_ids)


# API 3: Đọc file và Trả về 20 Câu hỏi Quiz theo cấp độ
@app.route("/get-quiz-questions", methods=["GET"])
def get_quiz_questions():
    """
    Đọc file JSON chứa câu hỏi quiz và trả về 20 câu hỏi ngẫu nhiên theo cấp độ.
    """
    # 1. Lấy tham số 'level'
    level_str = request.args.get("level", "1")
    try:
        requested_level = int(level_str)
        if requested_level < 1 or requested_level > 5:
            return jsonify({"error": "Cấp độ phải là số nguyên từ 1 đến 5."}), 400
    except ValueError:
        return jsonify({"error": "Tham số 'level' không hợp lệ. Vui lòng cung cấp số nguyên."}), 400

    # 2. Định nghĩa tên file và số lượng câu hỏi cần trả về
    QUIZ_FILE_NAME = "cooking_quiz_questions.json"
    N_QUESTIONS_RETURN = 20

    # 3. Đọc nội dung file JSON
    try:
        with open(QUIZ_FILE_NAME, 'r', encoding='utf-8') as f:
            all_questions = json.load(f)
            print(f">>> Đã tải {len(all_questions)} câu hỏi từ '{QUIZ_FILE_NAME}' <<<")

    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file quiz: {QUIZ_FILE_NAME}")
        return jsonify({"error": f"Lỗi: Không tìm thấy file câu hỏi {QUIZ_FILE_NAME}."}), 500
    except json.JSONDecodeError:
        print(f"LỖI: Định dạng JSON trong file {QUIZ_FILE_NAME} không hợp lệ.")
        return jsonify({"error": f"Lỗi: Định dạng JSON trong file {QUIZ_FILE_NAME} không hợp lệ."}), 500

    # 4. Lọc câu hỏi theo cấp độ yêu cầu
    filtered_questions = [
        q for q in all_questions if q.get("level") == requested_level
    ]
    print(f"Đã lọc được {len(filtered_questions)} câu hỏi ở Cấp độ {requested_level}.")

    if not filtered_questions:
        return jsonify({"error": f"Không tìm thấy câu hỏi nào cho Cấp độ {requested_level}."}), 404

    # 5. Chọn ngẫu nhiên N_QUESTIONS_RETURN câu hỏi
    final_quiz_set = random.sample(
        filtered_questions,
        min(N_QUESTIONS_RETURN, len(filtered_questions))
    )
    print(f"Đã chọn ngẫu nhiên {len(final_quiz_set)} câu hỏi cho Cấp độ {requested_level}.")

    # 6. Trả về kết quả
    return jsonify(final_quiz_set), 200

# API Ping
@app.route("/")
def hello():
    return "Chào! Server AI (Python Hybrid Microservice v2 - Anti-Stale) đang chạy!"

# =======================================================================
# CHẠY SERVER (CHỈ DÙNG CHO PHÁT TRIỂN CỤC BỘ)
# LƯU Ý: Khi deploy bằng Gunicorn, phần này sẽ bị bỏ qua.
# =======================================================================
if __name__ == "__main__":
    # Thay thế bằng os.environ.get('PORT', 5002) nếu deploy trên môi trường linh hoạt
    app.run(debug=True, port=5002)