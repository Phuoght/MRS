from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
from flask_limiter.util import get_remote_address
import spacy
import pandas as pd
import pickle
import numpy as np
from xgboost import XGBClassifier
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.preprocessing import normalize
import re
import logging
from html import escape

app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Danh sách stop words tiếng Việt
STOP_WORDS = {
    "là", "của", "và", "thì", "một", "được", "trong", "cho", "với", "tại",
    "bởi", "đã", "đang", "sẽ", "cũng", "mà", "này", "kia", "ở", "từ", "đến",
    "như", "nhưng", "hay", "rất", "hơn", "lại", "vẫn", "chỉ", "vào", "ra",
    "lên", "xuống", "gì", "thế", "nào", "ai", "đây", "đó", "khi", "nếu", "vì", "đi",
    "giúp tôi", "nhé", "nhiều", "những", "là", "không", "thì",
    "có thể", "được", "đã", "đang", "sẽ", "cũng", "mà", "này", "kia", "ở"
}

# Tải các mô hình
try:
    ner_model = spacy.load("models/model_ner_NVEB")
    intent_model = XGBClassifier()
    intent_model.load_model("models/intent_recognition_model.json")
    tokenizer = AutoTokenizer.from_pretrained("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")
    embedding_model = AutoModel.from_pretrained("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")
    with open('models/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    logger.error(f"Lỗi khi tải mô hình: {e}")
    exit(1)

# Tải dữ liệu
try:
    df_movies = pd.read_csv("data/data-film-final.csv")
    logger.info(f"Cột ban đầu của df_movies: {list(df_movies.columns)}")
    
    # Kiểm tra cột title
    if 'title' not in df_movies.columns:
        possible_title_cols = [col for col in df_movies.columns if col.lower() in ['name', 'movie_name', 'film_name']]
        if possible_title_cols:
            df_movies = df_movies.rename(columns={possible_title_cols[0]: 'title'})
            logger.info(f"Đổi tên cột {possible_title_cols[0]} thành 'title'")
        else:
            logger.error("Không tìm thấy cột 'title' hoặc cột tương tự trong dữ liệu")
            raise ValueError("Dữ liệu thiếu cột 'title'")
    
    # Chuẩn hóa dữ liệu
    df_movies['title'] = df_movies['title'].astype(str).str.strip().str.lower()
    df_movies['director'] = df_movies['director'].astype(str).str.strip().str.lower()
    df_movies['year'] = df_movies['release_year'].astype(str)
    df_movies['actor'] = df_movies['actor'].astype(str).str.strip().str.lower()
    df_movies['actor'] = df_movies['actor'].replace('', np.nan).fillna('không rõ')
    df_movies["actor_list"] = df_movies["actor"].apply(lambda x: [a.strip() for a in x.split(", ")] if x != 'không rõ' else [])
    df_movies['genre'] = df_movies['genre'].astype(str).str.strip().str.lower().apply(lambda x: x.replace("phim ", ""))
    df_movies['country'] = df_movies['country'].astype(str).str.strip().str.lower().apply(lambda x: x.replace("phim ", ""))
    df_movies['describe'] = df_movies['describe'].astype(str).str.strip().str.lower().apply(lambda x: x.replace("\xa0", ""))
except Exception as e:
    logger.error(f"Lỗi khi tải dữ liệu: {e}")
    exit(1)

# Tạo df_for_actor với cột title
try:
    df_for_actor = df_movies[['title', 'director', 'year', 'actor', 'actor_list', 'genre', 'describe', 'rating']].explode("actor_list").reset_index(drop=True)
    df_for_actor['actor_list'] = df_for_actor['actor_list'].str.strip().str.lower()
    df_for_actor = df_for_actor[df_for_actor['actor_list'] != ''].copy()
    logger.info(f"Cột của df_for_actor: {list(df_for_actor.columns)}")
except Exception as e:
    logger.error(f"Lỗi khi tạo df_for_actor: {e}")
    exit(1)

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z0-9À-ỹà-ỹ\s]", "", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    filtered_words = [word for word in words if word not in STOP_WORDS]
    return " ".join(filtered_words)

def predict_intent(text):
    text = preprocess_text(text)
    if not text:
        return "khac"
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            outputs = embedding_model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embedding = normalize(embedding, norm="l2")
        input_scaled = scaler.transform(embedding)
        pred = intent_model.predict(input_scaled)[0]
        return label_encoder.inverse_transform([pred])[0]
    except Exception as e:
        logger.error(f"Lỗi dự đoán ý định: {e}")
        return "khac"

def get_entities(text):
    text = preprocess_text(text)
    text = re.sub(r'(\b\d{4}\b)', r'năm \1', text)
    doc = ner_model(text)
    entities = {ent.label_: ent.text for ent in doc.ents}
    return entities

def generate_html(movie):
    if not isinstance(movie, pd.Series):
        return "Không có thông tin phim."
    title = movie.get('title', 'Không rõ')
    return (f'<div class="movie-card">'
            f'<b>{title.upper()} ({int(float(movie.get("year", 0)))})</b><br>'
            f'   ● <b>Thể loại</b>: {movie.get("genre", "Không rõ")}<br>'
            f'   ● <b>Rating</b>: {movie.get("rating", "Không rõ")}<br>' 
            f'   ● <b>Đạo diễn</b>: {movie.get("director", "Không rõ")}<br>'
            f'   ● <b>Quốc gia</b>: {movie.get("country", "Không rõ")}<br>'
            f'   ● <b>Diễn viên</b>: {movie.get("actor", "Không rõ")}<br>'
            f'   ● <b>Mô tả</b>: {movie.get("describe", "Không rõ")}<br>'
            f'</div>')

def suggest_movies(entities, df, criteria):
    if 'title' not in df.columns:
        logger.error(f"DataFrame đầu vào thiếu cột 'title': {df.columns}")
        return pd.DataFrame()
    logger.info(f"Đầu vào suggest_movies, cột của df: {list(df.columns)}")
    result = df.copy()
    for key, column in criteria.items():
        if key in entities:
            value = entities[key]
            if value:
                logger.info(f"Tìm kiếm {key} với giá trị: {value} trong cột {column}")
                try:
                    if key != 'rating_level' and column != 'rating':
                        result = result[result[column].str.contains(value, case=False, na=False)]
                    if key == 'rating_level' and column == 'rating':
                        if 'cao' in value:
                            result = result[result[column].fillna(0) >= 9.0]
                        elif 'tốt' in value:
                            result = result[(result["rating"].fillna(0) >= 7.0)]
                        else:
                            result = result[(result[column].fillna(0) >= 5.0) & (result[column].fillna(0) < 7.0)]
                    logger.info(f"Tìm thấy {len(result)} phim khớp với {key} = {value}. Cột: {list(result.columns)}, Tiêu đề: {result['title'].tolist()}")
                except Exception as e:
                    logger.error(f"Lỗi khi lọc theo {key}: {e}")
                    result = pd.DataFrame()
    return result


def update_temp_data(suggested_movies, df_movies_temp, df_for_actor_temp):
    if not suggested_movies.empty:
        if 'title' not in suggested_movies.columns:
            logger.error(f"suggested_movies thiếu cột 'title': {suggested_movies.columns}")
            return df_movies_temp, df_for_actor_temp
        # Chỉ xóa 5 phim đầu tiên
        titles = suggested_movies['title'].head(5).tolist()
        logger.info(f"Xóa 5 phim đầu tiên đã gợi ý: {titles}")
        df_movies_temp = df_movies_temp[~df_movies_temp['title'].isin(titles)]
        if 'title' not in df_for_actor_temp.columns:
            logger.error(f"df_for_actor_temp thiếu cột 'title' trước khi cập nhật: {df_for_actor_temp.columns}")
            df_for_actor_temp = df_for_actor.copy()
        else:
            df_for_actor_temp = df_for_actor_temp[~df_for_actor_temp['title'].isin(titles)]
    return df_movies_temp, df_for_actor_temp

def is_continue_request(text):
    text = preprocess_text(text)
    continue_keywords = ["tiếp tục", "tiếp tục đi", "tiếp", "thêm", "gợi ý thêm", "thêm phim", "thêm đi", "nữa", "tiếp tục nào"]
    return any(keyword in text for keyword in continue_keywords)

def compare_movies(entities):
    title1 = entities.get("title1")
    title2 = entities.get("title2")

    if title1 and title2:
        movie1 = df_movies[df_movies['title'] == title1]
        movie2 = df_movies[df_movies['title'] == title2]
        better = movie1 if movie1['rating'] > movie2['rating'] else movie2
        return (f"Đây là 2 bộ phim mà bạn đã đề cập tới:<br>"
                f"{generate_html(movie1)}<br>{generate_html(movie2)}<br>"
                f"Phim có đánh giá cao hơn là:<br>{better['title']}")
    return "Không thể so sánh vì không tìm thấy đủ thông tin. Thử lại với hai phim khác nhé!"

def search_movies(entities):
    keyword = entities.get("keywords") or entities.get("title")
    if keyword:
        return df_movies[df_movies['describe'].str.contains(keyword, case=False, na=False)].head(5)
    return pd.DataFrame()
def goi_y_theo_danh_gia_va_the_loai(genre, year, rating, df_movies_temp):
    if rating:
        if 'cao' in rating:
            filtered_movies = df_movies_temp[df_movies_temp["rating"].fillna(0) == 10.0]
        elif 'tốt' in rating:
            filtered_movies = df_movies_temp[(df_movies_temp["rating"].fillna(0) >= 8.0) & (df_movies_temp["rating"].fillna(0) < 10.0)]
        else:
            filtered_movies = df_movies_temp[(df_movies_temp["rating"].fillna(0) >= 5.0) & (df_movies_temp["rating"].fillna(0) < 8.0)]
    if genre:
        filtered_movies = df_movies_temp[
            (df_movies_temp["genre"].str.contains(genre, case=False, na=False))
        ]
    if year:
        filtered_movies = df_movies_temp[
            (df_movies_temp["year"].str.contains(year, case=False, na=False))
        ]
    result = filtered_movies.sort_values(by="rating", ascending=False).head(5)
    return result, "Đây là những phim đỉnh cao được đánh giá siêu tốt theo thể loại bạn chọn!" if genre else "Đây là những phim đỉnh cao được đánh giá siêu tốt!"

def handle_intent(intent, entities, user_input):
    global df_movies, df_for_actor

    # Khởi tạo dữ liệu tạm thời
    if 'df_movies_temp' not in session or 'df_for_actor_temp' not in session:
        session['df_movies_temp'] = df_movies.to_dict('records')
        session['df_for_actor_temp'] = df_for_actor.to_dict('records')
        session['chat_history'] = []
        logger.info("Khởi tạo session mới cho df_movies_temp và df_for_actor_temp")

    # Khôi phục df_movies_temp
    try:
        df_movies_temp = pd.DataFrame.from_dict(session['df_movies_temp'])
        if 'title' not in df_movies_temp.columns:
            logger.warning("Cột 'title' không tồn tại trong df_movies_temp, khởi tạo lại.")
            df_movies_temp = df_movies.copy()
            session['df_movies_temp'] = df_movies_temp.to_dict('records')
        logger.info(f"Cột của df_movies_temp sau khi khôi phục: {list(df_movies_temp.columns)}")
    except Exception as e:
        logger.error(f"Lỗi khi khôi phục df_movies_temp: {e}")
        df_movies_temp = df_movies.copy()
        session['df_movies_temp'] = df_movies_temp.to_dict('records')

    # Khôi phục df_for_actor_temp
    try:
        df_for_actor_temp = pd.DataFrame.from_dict(session['df_for_actor_temp'])
        if 'title' not in df_for_actor_temp.columns or 'actor_list' not in df_for_actor_temp.columns:
            logger.warning(f"Cột 'title' hoặc 'actor_list' không tồn tại trong df_for_actor_temp: {df_for_actor_temp.columns}, khởi tạo lại.")
            df_for_actor_temp = df_for_actor.copy()
            session['df_for_actor_temp'] = df_for_actor_temp.to_dict('records')
        df_for_actor_temp['actor_list'] = df_for_actor_temp['actor_list'].apply(
            lambda x: x.strip().lower() if isinstance(x, str) and x.strip() else 'không rõ'
        )
        logger.info(f"Cột của df_for_actor_temp sau khi khôi phục: {list(df_for_actor_temp.columns)}")
    except Exception as e:
        logger.error(f"Lỗi khi khôi phục df_for_actor_temp: {e}")
        df_for_actor_temp = df_for_actor.copy()
        session['df_for_actor_temp'] = df_for_actor_temp.to_dict('records')

    # Lưu lịch sử trò chuyện
    chat_history = session.get('chat_history', [])
    if len(chat_history) > 10:
        chat_history = chat_history[-10:]

    # Kiểm tra yêu cầu "tiếp tục"
    if is_continue_request(user_input) and chat_history:
        last_entry = chat_history[-1]
        last_intent = last_entry.get('intent')
        intent = last_intent
        entities = last_entry.get('entities', {})

    intent_handlers = {
        "goi_y_ket_hop": lambda: (suggest_movies(entities, df_movies_temp, {'genre': 'genre', 'year': 'year', 'director': 'director', 'rating_level': 'rating'}),
                                 "Tôi tìm được vài phim tuyệt vời cho bạn đây!"),
        "goi_y_theo_danh_gia": lambda: (goi_y_theo_danh_gia_va_the_loai(entities.get('genre', None),entities.get('year', None),entities.get('rating_level', None), df_movies_temp)),
        "goi_y_theo_dao_dien": lambda: (suggest_movies(entities, df_movies_temp, {'director': 'director', 'rating_level': 'rating'}),
                                       "Phim của đạo diễn này đúng là không thể bỏ lỡ!"),
        "goi_y_theo_dien_vien": lambda: (suggest_movies(entities, df_for_actor_temp, {'actor': 'actor_list', 'rating_level': 'rating'}),
                                        "Diễn viên này diễn đỉnh lắm, đây là vài phim hay cho bạn!"),
        "goi_y_theo_nam": lambda: (suggest_movies(entities, df_movies_temp, {'year': 'year', 'genre': 'genre', 'director': 'director', 'rating_level': 'rating'}),
                                  "Phim năm này có nhiều tác phẩm thú vị đây!"),
        "goi_y_theo_the_loai": lambda: (suggest_movies(entities, df_movies_temp, {'genre': 'genre', 'rating_level': 'rating'}),
                                       "Thể loại này đúng gu của bạn luôn, xem ngay nhé!"),
        "so_sanh_phim": lambda: (compare_movies(entities),
                                None),
        "tim_kiem_phim": lambda: (search_movies(entities),
                                 "Tôi tìm thấy vài phim liên quan đây, bạn ưng không?"),
        "khac": lambda: ("Hơi tiếc là tôi chưa hiểu rõ ý bạn, thử nói cụ thể hơn nhé?",
                        None)
    }

    handler = intent_handlers.get(intent, lambda: ("Ý định chưa được hỗ trợ, bạn thử lại nhé!", None))
    try:
        # Kiểm tra df_for_actor_temp trước khi gọi suggest_movies cho goi_y_theo_dien_vien
        if intent == "goi_y_theo_dien_vien":
            if 'title' not in df_for_actor_temp.columns:
                logger.error(f"df_for_actor_temp thiếu cột 'title' trước khi gọi suggest_movies: {df_for_actor_temp.columns}")
                df_for_actor_temp = df_for_actor.copy()
                session['df_for_actor_temp'] = df_for_actor_temp.to_dict('records')
                logger.info(f"Đã tái tạo df_for_actor_temp, cột mới: {list(df_for_actor_temp.columns)}")

        result, emotion_prefix = handler()
    except Exception as e:
        logger.error(f"Lỗi khi xử lý ý định: {e}")
        return {"user_input": user_input, "bot_response": "Có lỗi xảy ra khi xử lý yêu cầu, vui lòng thử lại!"}

    # Xử lý kết quả và cập nhật dữ liệu tạm thời
    if isinstance(result, pd.DataFrame):
        if not result.empty:
            try:
                if 'title' not in result.columns:
                    logger.error(f"DataFrame result thiếu cột 'title': {result.columns}")
                    reply = f"Dữ liệu phim không có tiêu đề, có thể do lỗi xử lý. Bạn thử lại nhé!"
                else:
                    logger.info(f"Phim được gợi ý: {result['title'].tolist()}")
                    reply = f"{emotion_prefix}<br>" + "<br>".join([generate_html(row) for _, row in result.head(5).iterrows()])
                    df_movies_temp, df_for_actor_temp = update_temp_data(result, df_movies_temp, df_for_actor_temp)
                    # Kiểm tra nếu hết phim
                    if intent == "goi_y_theo_dien_vien" and df_for_actor_temp.empty:
                        reply += "<br>Hết phim để gợi ý, bạn muốn thử tiêu chí khác không?"
                    elif intent in ["goi_y_ket_hop", "goi_y_theo_danh_gia", "goi_y_theo_dao_dien",
                                    "goi_y_theo_nam", "goi_y_theo_the_loai"] and df_movies_temp.empty:
                        reply += "<br>Hết phim để gợi ý, bạn muốn thử tiêu chí khác không?"
            except Exception as e:
                logger.error(f"Lỗi khi tạo HTML cho kết quả: {e}")
                reply = "Có lỗi khi hiển thị phim, bạn thử lại nhé!"
        else:
            if intent == "goi_y_theo_dien_vien" and 'actor' in entities:
                reply = f"Không tìm thấy phim của diễn viên {entities['actor']}. Bạn muốn thử ai khác không?"
            else:
                reply = "Ôi, tiếc quá! Tôi không tìm thấy phim phù hợp với yêu cầu này. Bạn muốn thử tiêu chí khác không?"
    else:
        reply = f"{emotion_prefix or ''} {result}" if emotion_prefix else result

    # Cập nhật lịch sử trò chuyện
    chat_history.append({"user_input": user_input, "intent": intent, "entities": entities})
    session['chat_history'] = chat_history

    # Lưu lại dữ liệu tạm thời vào session
    session['df_movies_temp'] = df_movies_temp.to_dict('records')
    session['df_for_actor_temp'] = df_for_actor_temp.to_dict('records')
    logger.info("Đã lưu session cho df_movies_temp và df_for_actor_temp")

    return {"user_input": user_input, "bot_response": reply}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("user_input")
    if not user_input or not isinstance(user_input, str):
        return jsonify({"user_input": "", "bot_response": "Dữ liệu đầu vào không hợp lệ."}), 400
    user_input = escape(user_input)
    intent = predict_intent(user_input)
    entities = get_entities(user_input)
    print(entities)
    result = handle_intent(intent, entities, user_input)
    return jsonify(result)

@app.route("/clear", methods=["POST"])
def clear():
    session.pop('df_movies_temp', None)
    session.pop('df_for_actor_temp', None)
    session.pop('chat_history', None)
    logger.info("Đã xóa session")
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    app.run(debug=True)