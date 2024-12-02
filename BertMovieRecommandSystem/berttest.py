import torch
from transformers import BertForSequenceClassification, BertTokenizer, BertModel
import pandas as pd
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity


# 1. 모델 로드 함수
def load_trained_model(model_path, num_labels=4):
    model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=num_labels)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device


# 2. BERT 기반 텍스트 임베딩 함수
def embed_text(text, embed_model, tokenizer, device, max_len=128):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = embed_model(**inputs)

    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()


# 3. 입력 데이터 변환 함수
def convert_input_data(sentences, tokenizer, max_len=128):
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=max_len, dtype="long", truncating="post", padding="post")
    attention_masks = [[float(i > 0) for i in seq] for seq in input_ids]
    inputs = torch.tensor(input_ids)
    masks = torch.tensor(attention_masks)
    return inputs, masks


# 4. 문장 예측 및 분류 함수
def test_sentences_with_titles(titles, sentences, model, tokenizer, device):
    inputs, masks = convert_input_data(sentences, tokenizer)
    inputs, masks = inputs.to(device), masks.to(device)

    with torch.no_grad():
        outputs = model(inputs, token_type_ids=None, attention_mask=masks)

    logits = outputs[0].detach().cpu().numpy()
    predictions = np.argmax(logits, axis=1)

    positive_titles, funny_titles, scary_titles = [], [], []

    for i, prediction in enumerate(predictions):
        if prediction == 1:  # 긍정
            positive_titles.append(titles[i])
        elif prediction == 2:  # 웃음
            funny_titles.append(titles[i])
        elif prediction == 3:  # 슬픔
            scary_titles.append(titles[i])

    return positive_titles, funny_titles, scary_titles


# 5. 추천 함수
def recommend_similar_movies(selected_titles, all_titles, all_embeddings, top_k=5):
    selected_embeddings = [all_embeddings[all_titles.index(title)] for title in selected_titles if title in all_titles]
    if not selected_embeddings:
        return []

    mean_embedding = np.mean(selected_embeddings, axis=0).reshape(1, -1)
    all_embeddings = np.vstack(all_embeddings)
    similarities = cosine_similarity(mean_embedding, all_embeddings).flatten()
    sorted_indices = similarities.argsort()[::-1]

    recommendations = []
    for idx in sorted_indices:
        title = all_titles[idx]
        if title not in selected_titles:
            recommendations.append(title)
            if len(recommendations) >= top_k:
                break
    return recommendations


# 6. 메인 실행 흐름
def main():
    # 모델 및 토크나이저 로드
    sentiment_model_path = 'final_model02.bin'
    sentiment_model, device = load_trained_model(sentiment_model_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    embed_model = BertModel.from_pretrained('bert-base-multilingual-cased')
    embed_model.to(device)

    # 영화 데이터 로드 및 임베딩 생성
    movies = pd.read_csv('tmdb_5000_movies.csv')
    all_titles = movies['title'].tolist()
    all_overviews = movies['overview'].fillna("").tolist()
    all_embeddings = [embed_text(overview, embed_model, tokenizer, device) for overview in all_overviews]

    # 감정 분석
    test_titles = ['Titanic', 'Iron Man', 'Me Before You', 'The Hangover']
    test_sentences = ['감동적이고 아름다운 영화입니다.', '지루했지만 괜찮았어요.', '너무 슬퍼서 펑펑 울었다ㅜ', '진짜 웃겨서 배꼽빠지는 줄 알았다ㅋㅋㅋ']
    positive_titles, funny_titles, scary_titles = test_sentences_with_titles(
        test_titles, test_sentences, sentiment_model, tokenizer, device)

    print("긍정으로 예측된 영화:", positive_titles)
    print("웃음으로 예측된 영화:", funny_titles)
    print("슬픔으로 예측된 영화:", scary_titles)

    # 영화 추천
    recommended_positive = recommend_similar_movies(positive_titles, all_titles, all_embeddings, top_k=5)
    recommended_funny = recommend_similar_movies(funny_titles, all_titles, all_embeddings, top_k=5)
    recommended_scary = recommend_similar_movies(scary_titles, all_titles, all_embeddings, top_k=5)

    print("긍정 영화 추천:", recommended_positive)
    print("웃음 영화 추천:", recommended_funny)
    print("슬픈 영화 추천:", recommended_scary)


if __name__ == "__main__":
    main()
