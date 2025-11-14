# --- File: app_ai_service.py (Python AI Service) ---
# SERVER PYTHON CHUYÊN BIỆT (VI DỊCH VỤ)
# Chức năng: Gợi ý Hybrid, Gắn Tag NLP, Tạo Quiz Tùy chỉnh.

import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from datetime import datetime
import os
import json
import random

# Import thư viện Firebase Admin
import firebase_admin
from firebase_admin import credentials, firestore

# Import thư viện Gemini
from google import genai
from google.genai import types

# =======================================================================
# KHỞI TẠO ỨNG DỤNG VÀ FIREBASE
# =======================================================================
app = Flask(__name__)
db = None
GEMINI_CLIENT = None

# --- Khởi tạo Firebase ---
try:
    key_json_str = os.environ.get("FIREBASE_KEY_JSON")

    if key_json_str:
        key_json_dict = json.loads(key_json_str)
        print(">>> Đang dùng khóa từ Biến Môi Trường/ .env <<<")
    else:
        try:
            with open("serviceAccountKey.json", 'r') as f:
                key_json_dict = json.load(f)
            print(">>> Đang dùng khóa từ file 'serviceAccountKey.json' (Fallback) <<<")
        except FileNotFoundError:
            raise Exception("Lỗi: Không tìm thấy khóa Firebase.")

    cred = credentials.Certificate(key_json_dict)
    firebase_admin.initialize_app(cred, {'projectId': 'mealmaker-backend'})

    db = firestore.client()
    print(">>> KẾT NỐI FIREBASE (FIRESTORE) THÀNH CÔNG! <<<")

except Exception as e:
    print(f"LỖI: Không thể kết nối Firebase. Lỗi: {e}")

# --- Khởi tạo Gemini ---
try:
    # Client sẽ tự động đọc GEMINI_API_KEY từ biến môi trường
    GEMINI_CLIENT = genai.Client()
    print(">>> KẾT NỐI GEMINI CLIENT THÀNH CÔNG! <<<")
except Exception as e:
    print(f"CẢNH BÁO: Không thể khởi tạo Gemini Client. Vui lòng kiểm tra biến môi trường GEMINI_API_KEY. Lỗi: {e}")


# =======================================================================
# TẢI "BỘ NÃO" AI VÀO BỘ NHỚ
# =======================================================================
model_knn = None
user_item_matrix = None
recipe_id_map = None
user_id_map = None
model_tagger = None
vectorizer_tagger = None
mlb_tagger = None

model_prefix_knn = "ai_model_knn"
try:
    model_knn = joblib.load(f"{model_prefix_knn}_model.pkl")
    user_item_matrix = joblib.load(f"{model_prefix_knn}_matrix.pkl")
    recipe_id_map = joblib.load(f"{model_prefix_knn}_recipe_id_map.pkl")
    user_id_map = joblib.load(f"{model_prefix_knn}_user_id_map.pkl")
    print(f">>> TẢI 'BỘ NÃO AI k-NN' THÀNH CÔNG! (Đã huấn luyện {user_item_matrix.shape[0]} users) <<<")
except Exception as e:
    print(f"CẢNH BÁO: Không tải được 'bộ não' k-NN. ({e})")

model_prefix_tagger = "ai_tagger"
try:
    model_tagger = joblib.load(f"{model_prefix_tagger}_model.pkl")
    vectorizer_tagger = joblib.load(f"{model_prefix_tagger}_vectorizer.pkl")
    mlb_tagger = joblib.load(f"{model_prefix_tagger}_mlb.pkl")
    print(f">>> TẢI 'BỘ NÃO AI GẮN TAG' (NLP) THÀNH CÔNG! <<<")
except Exception as e:
    print(f"CẢNH BÁO: Không tải được 'bộ não' Gắn Tag. ({e})")


# =======================================================================
# ĐỊNH NGHĨA CẤP ĐỘ VÀ LOGIC TẠO QUIZ (AI CREATION LOGIC)
# =======================================================================
LEVEL_DESCRIPTIONS = {
    1: "Cơ bản: Nhận diện nguyên liệu, an toàn thực phẩm, sơ chế đơn giản (gọt, rửa).",
    2: "Trung cấp: Kỹ thuật thái, kiểm soát nhiệt độ chảo, nêm nếm cơ bản.",
    3: "Nâng cao: Kỹ thuật làm sốt, áp chảo chuyên sâu, khắc phục lỗi.",
    4: "Chuyên gia: Kỹ thuật nấu ăn phức tạp, hiểu biết cấu trúc thực phẩm, ẩm thực đặc thù.",
    5: "Siêu cấp: Kết hợp hương vị phức tạp, tối ưu hóa quy trình, sáng tạo công thức."
}
TOTAL_QUIZ_COUNT = 25

def generate_quiz_with_gemini(target_level, description, count):
    """
    Sử dụng Gemini API để tạo câu hỏi trắc nghiệm theo độ khó.
    """
    if not GEMINI_CLIENT:
        raise Exception("Gemini Client chưa được khởi tạo.")

    # Xây dựng Prompt QUAN TRỌNG: yêu cầu độ khó và định dạng JSON
    prompt = f"""
    TẠO BỘ CÂU HỎI TRẮC NGHIỆM KỸ NĂNG NẤU ĂN.

    Yêu cầu:
    1. Tạo chính xác {count} câu hỏi.
    2. Độ khó phải TUYỆT ĐỐI tuân thủ mô tả cấp độ: "Cấp độ {target_level}: {description}".
    3. Mỗi câu hỏi phải có 4 lựa chọn (A, B, C, D), trong đó có chính xác 1 đáp án đúng.

    YÊU CẦU ĐỊNH DẠNG:
    Trả lời bằng một đối tượng JSON Array. KHÔNG có bất kỳ văn bản giải thích nào khác ngoài JSON.
    Cấu trúc JSON cho mỗi câu hỏi:
    {{
      "level": {target_level},
      "question": (string),
      "options": [
        {{"id": "a", "text": (string)}},
      ],
      "correct_id": (string, ví dụ: "a", "b"),
      "explanation": (string, giải thích ngắn gọn tại sao đáp án đúng)
    }}
    """

    try:
        response = GEMINI_CLIENT.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            ),
        )
        return json.loads(response.text)

    except Exception as e:
        print(f"Lỗi khi gọi Gemini API để tạo quiz: {e}")
        return None


# =======================================================================
# API ENDPOINTS
# =======================================================================

# API 1: Dùng AI Huấn luyện để Gắn Tag
@app.route("/get-auto-tags", methods=["POST"])
def get_auto_tags():
    if not model_tagger:
        return jsonify({"error": "Bộ não AI Gắn Tag (NLP) chưa sẵn sàng."}), 500

    try:
        data = request.json
        title = data.get("title", "")
        ingredients_list = data.get("ingredients_list", [])
        instructions_list = data.get("instructions", [])
        time = data.get("time_minutes", 0)

        ingredients_str = " ".join(ingredients_list)
        instructions_str = " ".join(instructions_list)
        full_text = f"{title} {ingredients_str} {instructions_str} time_{time}"

        text_vector = vectorizer_tagger.transform([full_text])
        predicted_probabilities = model_tagger.predict_proba(text_vector)

        THRESHOLD = 0.3
        predicted_binary = (predicted_probabilities > THRESHOLD).astype(int)
        predicted_tags = mlb_tagger.inverse_transform(predicted_binary)
        final_tags = list(predicted_tags[0])

        # Logic cứng cho tag thời gian
        TIME_TAGS = ["quick", "medium_cook", "long_cook"]
        final_tags = [tag for tag in final_tags if tag not in TIME_TAGS]

        time_tag_rule = None
        if time > 0 and time <= 25:
            time_tag_rule = "quick"
        elif time > 25 and time <= 60:
            time_tag_rule = "medium_cook"
        elif time > 60:
            time_tag_rule = "long_cook"

        if time_tag_rule:
            final_tags.append(time_tag_rule)

        return jsonify(final_tags), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API 2: Gợi ý (Hybrid AI)
@app.route("/get-recommendations-ai", methods=["GET"])
def get_recommendations_ai():
    # ... (Logic k-NN và Fallback dựa trên ai_profile giữ nguyên như trong code ban đầu) ...
    # Chú ý: Phần này bị lược bớt để file tổng hợp gọn hơn, nhưng bạn cần giữ nguyên logic đã được tối ưu trước đó
    if not db: return jsonify({"error": "Lỗi CSDL"}), 500
    if not model_knn: return jsonify({"error": "Bộ não AI k-NN chưa sẵn sàng."}), 500

    user_id = request.args.get("userId")
    if not user_id:
        return jsonify({"error": "Cần cung cấp userId"}), 400

    recommendation_ids = []
    K_FINAL = 14

    # --- 1. k-NN Logic (Giả sử thành công) ---
    if user_id in user_id_map:
        # (Thực hiện logic k-NN, sau đó gán kết quả vào recommendation_ids)
        # Vì đây là ví dụ tổng hợp, ta giả định logic k-NN phức tạp đã chạy
        if not recommendation_ids:
             # Logic k-NN thất bại (ví dụ: không tìm thấy user tương đồng), chuyển sang fallback.
             pass

    # --- 2. FALLBACK (DỰ PHÒNG) ---
    if not recommendation_ids:
        try:
            user_doc = db.collection("users").document(user_id).get()
            if user_doc.exists and "ai_profile" in user_doc.to_dict():
                ai_profile = user_doc.to_dict()["ai_profile"]
                tags_to_query = [*(ai_profile.get("diet", [])), *(ai_profile.get("favorite_cuisines", []))]

                if tags_to_query:
                    recommendations_ref = db.collection("recipes").where("tags", "array-contains-any", tags_to_query).limit(50).stream()
                    fallback_potential_ids = [r.to_dict().get("recipe_id") for r in recommendations_ref]

                    if fallback_potential_ids:
                        random.shuffle(fallback_potential_ids)
                        recommendation_ids = fallback_potential_ids[:K_FINAL]

        except Exception as e:
            print(f"Lỗi khi dùng Fallback (Habits): {e}")
            pass

    if not recommendation_ids:
        return jsonify({"error": "Không tìm thấy gợi ý nào"}), 404

    return jsonify(recommendation_ids)


# API 3: Tạo Bộ Đề Trắc Nghiệm TÙY CHỈNH theo Cấp độ
@app.route("/generate-skill-quiz", methods=["GET"])
def generate_skill_quiz():
    if not db: return jsonify({"error": "Lỗi CSDL"}), 500
    if not GEMINI_CLIENT: return jsonify({"error": "Lỗi AI Service: Gemini chưa sẵn sàng."}), 500

    user_id = request.args.get("userId")
    if not user_id:
        return jsonify({"error": "Cần cung cấp userId"}), 400

    try:
        user_doc = db.collection("users").document(user_id).get()
        user_data = user_doc.to_dict() if user_doc.exists else {}

        # Lấy cấp độ kỹ năng hiện tại (Mặc định: 1)
        current_level = user_data.get("ai_profile", {}).get("cooking_skill_level", 1)
        current_level = max(1, min(5, int(current_level)))

        level_description = LEVEL_DESCRIPTIONS.get(current_level)

        # GỌI HÀM GEMINI THỰC TẾ: Tạo 25 câu tập trung vào Cấp độ hiện tại
        all_questions = generate_quiz_with_gemini(
            target_level=current_level,
            description=level_description,
            count=TOTAL_QUIZ_COUNT
        )

        if not all_questions or len(all_questions) == 0:
             return jsonify({"error": "AI không thể tạo câu hỏi. Vui lòng thử lại."}), 500

        # Gán ID duy nhất và trộn ngẫu nhiên
        final_questions = []
        for i, q in enumerate(all_questions):
            q['id'] = i + 1
            final_questions.append(q)

        random.shuffle(final_questions)

        return jsonify(final_questions), 200

    except Exception as e:
        print(f"Lỗi chung khi tạo bộ đề trắc nghiệm: {e}")
        return jsonify({"error": str(e)}), 500

# API 4: Lấy các Chủ đề Cần Cải thiện (Skill Keys)
@app.route("/get-skill-improvement-keys", methods=["GET"])
def get_skill_improvement_keys():
    if not db: return jsonify({"error": "Lỗi CSDL"}), 500

    user_id = request.args.get("userId")
    top_n = int(request.args.get("topN", 3))

    if not user_id:
        return jsonify({"error": "Cần cung cấp userId"}), 400

    try:
        user_doc = db.collection("users").document(user_id).get()
        user_data = user_doc.to_dict()

        skill_profile = user_data.get("skill_profile")

        if not skill_profile or "topic_scores" not in skill_profile:
            # Trả về các chủ đề mặc định nếu chưa có dữ liệu làm bài
            return jsonify(["SơChế", "AnToanThucPham", "KiemSoatNhiet"])

        topic_scores = skill_profile["topic_scores"]

        scores_list = [(key, score) for key, score in topic_scores.items()]

        # Sắp xếp TĂNG DẦN (tìm những chủ đề điểm thấp nhất)
        scores_list.sort(key=lambda item: item[1])

        # Lấy Top N chủ đề yếu nhất
        improvement_keys = [item[0] for item in scores_list[:top_n]]

        return jsonify(improvement_keys), 200

    except Exception as e:
        print(f"Lỗi khi lấy Skill Improvement Keys: {e}")
        return jsonify({"error": str(e)}), 500


# API Ping
@app.route("/")
def hello():
    return "Chào! Server AI (Python Hybrid Microservice Final) đang chạy!"

# =======================================================================
# CHẠY SERVER
# =======================================================================
if __name__ == "__main__":
    app.run(debug=True, port=5002)