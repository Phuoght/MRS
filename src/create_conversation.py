import random
import pandas as pd
from collections import defaultdict
import json
from itertools import cycle

def generate_training_data_final_optimized(df, config, target_intent, seen_texts_global, seen_texts_local=None):
    intents_config = config.get("intents", {})
    
    def normalize(text):
        return str(text).lower().strip().replace('\xa0', ' ')
    
    entities_from_df = {
        "title": df['title'].apply(normalize).tolist(),
        "genre": df['genre'].apply(normalize).apply(lambda x: x.replace('phim ', '')).dropna().unique().tolist(),
        "actor": list({a.strip().lower() for actors in df['actor'].dropna() for a in actors.split(',') if a.strip()}),
        "director": df['director'].apply(normalize).dropna().unique().tolist(),
        "year": df['release_year'].dropna().astype(int).astype(str).unique().tolist(),
        "rating": df['rating'].dropna().unique().tolist() if 'rating' in df.columns else [str(i) for i in range(1, 11)],
        "describe": df['describe'].apply(normalize).dropna().tolist(),
    }
    
    entity_types = defaultdict(list)
    entity_types["title"] = entities_from_df["title"]
    entity_types["title1"] = entities_from_df["title"]
    entity_types["title2"] = entities_from_df["title"]
    entity_types["genre"] = list(set(entities_from_df["genre"]))
    entity_types["actor"] = entities_from_df["actor"]
    entity_types["director"] = entities_from_df["director"]
    entity_types["year"] = entities_from_df["year"]
    entity_types["rating_level"] = ["cao", "tốt", "trên trung bình"]
    entity_types["min_rating"] = sorted([str(r) for r in entities_from_df["rating"]], reverse=True)
    entity_types["keywords"] = entities_from_df["describe"]
    entity_types["another_actor"] = entities_from_df["actor"]
    entity_types["actor1"] = entities_from_df["actor"]
    entity_types["actor2"] = entities_from_df["actor"]
    entity_types["country"] = entities_from_df["country"]
    
    # In số lượng thực thể để kiểm tra
    print(f"Title: {len(set(entities_from_df['title']))}")
    print(f"Genre: {len(set(entities_from_df['genre']))}")
    print(f"Actor: {len(set(entities_from_df['actor']))}")
    print(f"Director: {len(set(entities_from_df['director']))}")
    print(f"Year: {len(set(entities_from_df['year']))}")
    
    training_data = []
    seen_texts = seen_texts_global
    if seen_texts_local is None:
        seen_texts_local = set()  # Cho phép trùng lặp trong cùng tập
    
    def generate_sample(intent, templates, entities_needed, entity_types, seen_texts, seen_texts_local):
        template_cycle = cycle(templates)
        max_attempts = 1000
        attempts = 0
        
        while attempts < max_attempts:
            template = next(template_cycle)
            entity_values = {}
            for entity in entities_needed:
                values = entity_types.get(entity, [])
                if not values:
                    return None
                entity_values[entity] = random.choice(values)
            
            try:
                formatted_text = template.format(**entity_values)
                if formatted_text not in seen_texts:  # Chỉ kiểm tra trùng lặp giữa các tập
                    seen_texts.add(formatted_text)
                    seen_texts_local.add(formatted_text)
                    extracted_entities = []
                    for entity, value in entity_values.items():
                        idx = formatted_text.find(value)
                        if idx != -1:
                            extracted_entities.append({
                                "entity": entity,
                                "value": value,
                                "start": idx,
                                "end": idx + len(value)
                            })
                    return {
                        "text": formatted_text,
                        "intent": intent,
                        "entities": extracted_entities
                    }
                elif formatted_text not in seen_texts_local:  # Cho phép trùng trong cùng tập
                    seen_texts_local.add(formatted_text)
                    extracted_entities = []
                    for entity, value in entity_values.items():
                        idx = formatted_text.find(value)
                        if idx != -1:
                            extracted_entities.append({
                                "entity": entity,
                                "value": value,
                                "start": idx,
                                "end": idx + len(value)
                            })
                    return {
                        "text": formatted_text,
                        "intent": intent,
                        "entities": extracted_entities
                    }
            except (KeyError, ValueError):
                pass
            attempts += 1
        return None
    
    all_intents = list(intents_config.keys()) + ["khac"]
    
    for intent in all_intents:
        intent_samples = []
        if intent == "khac":
            noise_types = [
                "random_words", "other_subjects", "mixed_entities", "irrelevant_keywords",
                "ambiguous_questions", "random_grammar", "noise_injection", "misspelling", "truncated",
                "random_phrase"
            ]
            all_entity_keys = list(entity_types.keys())
            
            def generate_random_grammar_sentence():
                grammar = {
                    "S": ["NP VP .", "NP VP ADVP .", "VP NP ."],
                    "NP": ["DT NN", "PRP", "NN NN", "DT ADJP NN"],
                    "VP": ["VB", "VB ADJP", "VB PP", "VB NP"],
                    "ADVP": ["RB", "RB ADJP"],
                    "PP": ["IN NP"],
                    "DT": ["một", "cái", "những", "các", "mấy"],
                    "NN": ["ý tưởng", "câu hỏi", "thông tin", "thứ", "việc", "người", "trò chơi", "món ăn"],
                    "PRP": ["tôi", "bạn", "nó", "họ", "chúng tôi"],
                    "VB": ["nghe", "thấy", "biết", "làm", "hỏi", "tìm", "chơi", "ăn"],
                    "ADJP": ["lạ", "vô nghĩa", "khó hiểu", "tuyệt vời", "hay", "dễ", "đẹp"],
                    "RB": ["rất", "hơi", "khá", "chưa", "luôn"],
                    "IN": ["về", "ở", "tại", "với", "trong"]
                }
                def generate(symbol):
                    if symbol in grammar:
                        return " ".join(generate(s) for s in random.choice(grammar[symbol]).split())
                    return symbol
                return generate("S")
            
            while len(intent_samples) < target_intent:
                negative_type = random.choice(noise_types)
                text = None
                available_samples = training_data + intent_samples
                if negative_type == "random_words":
                    phrase = " ".join(random.choice(
                        ["phim", "ảnh", "nhạc", "sách", "thời tiết", "tin tức", "bóng đá", "du lịch", "nấu ăn", "học tập", "máy bay", "xe hơi", "trò chơi"]
                    ) for _ in range(random.randint(3, 8)))
                    text = phrase + "."
                elif negative_type == "other_subjects":
                    text = random.choice([
                        "thời tiết hôm nay thế nào", "bài hát mới nhất của ai", "tin tức nóng hổi là gì", 
                        "cách làm bánh", "du lịch ở đâu đẹp", "giá vàng hôm nay bao nhiêu", "trận bóng tối nay ra sao",
                        "trò chơi nào hay nhất", "món ăn nào ngon"
                    ])
                elif negative_type == "mixed_entities" and entity_types:
                    mixed_entities = random.sample(all_entity_keys, random.randint(1, min(3, len(all_entity_keys))))
                    mixed_values = " ".join(random.choice(entity_types[entity]) for entity in mixed_entities)
                    text = f"{mixed_values} thì sao"
                elif negative_type == "irrelevant_keywords":
                    phrase = " ".join(random.choice(
                        ["giá vàng", "chứng khoán", "tỷ giá", "luật giao thông", "sức khỏe", "thể thao", "công nghệ", "máy tính", "điện thoại"]
                    ) for _ in range(random.randint(2, 5)))
                    text = phrase + "."
                elif negative_type == "ambiguous_questions":
                    text = random.choice([
                        "tôi muốn xem cái gì đó hay", "gợi ý cho tôi một thứ gì đó", "có gì thú vị không", 
                        "tìm kiếm một thứ gì đó", "hôm nay có gì đặc biệt không", "làm gì bây giờ nhỉ"
                    ])
                elif negative_type == "random_grammar":
                    text = generate_random_grammar_sentence()
                elif negative_type == "noise_injection" and available_samples:
                    positive_sample = random.choice(available_samples)
                    text = positive_sample["text"]
                    noise_choice = random.random()
                    if noise_choice < 0.3:
                        if text:
                            idx = random.randint(0, len(text) - 1)
                            text = text[:idx] + random.choice("abcdefghijklmnopqrstuvwxyz") + text[idx+1:]
                    elif noise_choice < 0.6:
                        text += " " + random.choice(["tuyệt vời", "hôm nay", "bất ngờ", "có lẽ", "nhé", "đi"])
                    else:
                        if positive_sample["entities"]:
                            entity_to_replace = random.choice(positive_sample["entities"])
                            other_keys = [k for k in all_entity_keys if k != entity_to_replace["entity"] and entity_types.get(k)]
                            if other_keys:
                                new_key = random.choice(other_keys)
                                new_value = random.choice(entity_types[new_key])
                                text = text.replace(entity_to_replace["value"], new_value, 1)
                elif negative_type == "misspelling":
                    word = random.choice(["phim", "đạo diễn", "diễn viên", "thể loại", "tìm", "xem", "gợi ý"])
                    misspelled = word[:len(word)//2] + random.choice("aeiou") + word[len(word)//2+1:]
                    text = f"tìm {misspelled} nào hay"
                elif negative_type == "truncated" and available_samples:
                    positive_sample = random.choice(available_samples)["text"]
                    if len(positive_sample) >= 10:
                        max_end = len(positive_sample) // 2
                        text = positive_sample[:random.randint(min(5, max_end), max_end)]
                    else:
                        text = positive_sample
                elif negative_type == "random_phrase":
                    subjects = ["ai đó", "cái gì", "ở đâu", "hôm nay", "ngày mai"]
                    verbs = ["làm", "xem", "nghe", "tìm", "nói", "hỏi"]
                    objects = ["gì đó", "một thứ", "cái hay", "bất ngờ", "vui vẻ"]
                    text = f"{random.choice(subjects)} {random.choice(verbs)} {random.choice(objects)}"
                
                if text and text not in seen_texts:
                    seen_texts.add(text)
                    seen_texts_local.add(text)
                    intent_samples.append({"text": text, "intent": "khac", "entities": []})
                elif text and text not in seen_texts_local:  # Cho phép trùng trong cùng tập
                    seen_texts_local.add(text)
                    intent_samples.append({"text": text, "intent": "khac", "entities": []})
        else:
            templates = intents_config[intent]["templates"]
            entities_needed = intents_config[intent]["entities"]
            while len(intent_samples) < target_intent:
                sample = generate_sample(intent, templates, entities_needed, entity_types, seen_texts, seen_texts_local)
                if sample:
                    intent_samples.append(sample)
                else:
                    print(f"Không thể sinh đủ mẫu cho intent '{intent}'. Đã sinh: {len(intent_samples)}/{target_intent}")
                    break
        
        training_data.extend(intent_samples)
    
    random.shuffle(training_data)
    
    intent_counts = defaultdict(int)
    for sample in training_data:
        intent_counts[sample["intent"]] += 1
    print(f"Số mẫu theo intent: {dict(intent_counts)}")
    
    return training_data

# Chạy code với config đã sửa
if __name__ == "__main__":
    df_movies = pd.read_csv('../data/data-film-final.csv')
    config = {
        "intents": {
            "goi_y_theo_the_loai": {
                "templates": [
                    "gợi ý phim {genre}",
                    "tôi muốn xem phim thể loại {genre}",
                    "tìm phim {genre} được đánh giá cao",
                    "khuyên tôi một bộ phim {genre}",
                    "phim thuộc thể loại {genre} nào hay",
                    "phim {genre} nào đang hot hiện nay",
                    "top 10 phim {genre} hay nhất",
                    "gợi ý phim {genre} cho người mới bắt đầu",
                    "phim {genre} có cốt truyện hấp dẫn",
                    "phim {genre} phù hợp cho gia đình",
                    "phim {genre} nào đáng xem",
                    "gợi ý phim {genre} nổi tiếng",
                    "tìm phim {genre} hay nhất mọi thời đại",
                    # Thêm template kết hợp title
                    "gợi ý phim {genre} như {title}",
                    "tìm phim {genre} giống {title}",
                    "phim {genre} hay như {title}"
                ],
                "entities": ["genre", "title"],
                "weight": 2
            },
            "goi_y_theo_dien_vien": {
                "templates": [
                    "phim nào của {actor} hay",
                    "gợi ý phim có {actor} đóng",
                    "tìm phim có sự tham gia của {actor}",
                    "cho tôi xem phim có {actor}",
                    "bộ phim nào có {actor} xuất hiện",
                    "phim nổi tiếng nhất của {actor}",
                    "phim mới nhất của {actor}",
                    "phim {actor} đóng vai chính",
                    "phim có {actor} và đạo diễn nổi tiếng",
                    "phim hành động có {actor}"
                ],
                "entities": ["actor"],
                "weight": 1.5
            },
            "goi_y_theo_dao_dien": {
                "templates": [
                    "phim của đạo diễn {director}",
                    "tìm phim do {director} đạo diễn",
                    "gợi ý phim {genre} của đạo diễn {director}",
                    "cho tôi biết các phim của {director}",
                    "bộ phim nào do {director} chỉ đạo"
                ],
                "entities": ["director", "genre"],
                "weight": 1.5
            },
            "so_sanh_phim": {
                "templates": [
                    "so sánh {title1} và {title2}",
                    "{title1} với {title2} khác nhau thế nào",
                    "điểm giống và khác nhau giữa {title1} và {title2}",
                    "phân biệt {title1} và {title2} giúp tôi",
                    "sự khác biệt giữa phim {title1} và {title2} là gì",
                    "nên xem {title1} hay {title2}",
                    "phim nào hay hơn giữa {title1} và {title2}",
                    "so sánh cốt truyện của {title1} và {title2}",
                ],
                "entities": ["title1", "title2"],
                "weight": 1
            },
            "tim_kiem_phim": {
                "templates": [
                    "tìm phim về {keywords}",
                    "tôi muốn xem phim có {keywords}",
                    "có phim {genre} nào về {keywords} không",
                    "cho tôi danh sách phim có {keywords}",
                    "đâu là bộ phim chứa từ {keywords}",
                    "phim nói về {keywords}",
                    "tìm phim có cảnh {keywords}",
                    "phim có nhân vật {keywords}",
                    "phim lấy bối cảnh ở {keywords}",
                    "phim có chủ đề {keywords}"
                ],
                "entities": ["keywords", "genre"],
                "weight": 1.2
            },
            "goi_y_theo_nam": {
                "templates": [
                    "phim năm {year}",
                    "gợi ý phim từ năm {year}",
                    "tìm phim {genre} năm {year}",
                    "cho tôi xem phim ra mắt năm {year}",
                    "bộ phim nào được sản xuất vào năm {year}",
                    "phim hay năm {year}",
                    "top phim của năm {year}",
                    "phim {genre} ra mắt năm {year}",
                    "phim nổi tiếng năm {year}",
                    "gợi ý phim hay từ {year}",
                    # Thêm template kết hợp title
                    "phim {genre} năm {year} giống {title}",
                    "gợi ý phim năm {year} như {title}"
                ],
                "entities": ["year", "genre", "title"],
                "weight": 0.8
            },
            "goi_y_theo_danh_gia": {
                "templates": [
                    "gợi ý phim được đánh giá {rating_level}",
                    "tìm phim rating từ {min_rating} trở lên",
                    "gợi ý phim {genre} được đánh giá {rating_level}",
                    "cho tôi biết phim nào có điểm đánh giá {min_rating} trở lên",
                    "phim {genre} có đánh giá {rating_level} là gì",
                    "phim có điểm IMDb từ {min_rating} trở lên",
                    "phim {genre} có rating cao",
                    "phim {genre} được đánh giá tốt",
                    "tìm phim hay có điểm {min_rating}",
                    "phim {genre} đánh giá {rating_level} giống {title}",
                    "tìm phim rating {min_rating} như {title}"
                ],
                "entities": ["rating_level", "min_rating", "genre", "title"],
                "weight": 0.8
            },
            "goi_y_ket_hop": {
                "templates": [
                    "phim {genre} của {director}",
                    "phim {genre} của {actor}",
                    "gợi ý phim {genre} năm {year} với {actor}",
                    "gợi ý phim {genre} năm {year} rating trên {min_rating}",
                    "gợi ý phim {genre} của {director}",
                    "gợi ý phim {genre} {country} năm {year}",
                    "tìm phim {genre} của {director}",
                    "phim {genre} {country}",
                    "gợi ý phim {genre} {country}",
                    "gợi ý phim {genre} {country} rating cao",
                    "gợi ý phim {genre} {country} năm {year} rating cao",
                ],
                "entities": ["genre", "director", "actor", "year", "min_rating", "country"],
                "weight": 1.5
            },
        }
    }
    
    seen_texts_global = set()
    
    # Train
    seen_texts_train = set()
    training_data = generate_training_data_final_optimized(df_movies, config, 7000, seen_texts_global, seen_texts_train)
    
    # Valid
    seen_texts_valid = set()
    valid_data = generate_training_data_final_optimized(df_movies, config, 2000, seen_texts_global, seen_texts_valid)
    
    # Test
    seen_texts_test = set()
    test_data = generate_training_data_final_optimized(df_movies, config, 1000, seen_texts_global, seen_texts_test)
    
    with open("../data/movie_training.json", "w", encoding="utf-8") as f:
        json.dump(training_data, f, ensure_ascii=False, indent=4)
    with open("../data/movie_validation.json", "w", encoding="utf-8") as f:
        json.dump(valid_data, f, ensure_ascii=False, indent=4)
    with open("../data/movie_test.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)
    
    print(f"\nTập huấn luyện: {len(training_data)} mẫu, tập validation: {len(valid_data)} mẫu, tập test: {len(test_data)} mẫu đã được lưu.")