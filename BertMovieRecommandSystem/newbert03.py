import tensorflow as tf
import torch

from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import random
import time
import datetime
import numpy as np

train = pd.read_csv("nsmcMulti01_train.csv", encoding="cp949")  # 또는 "cp949"
print(train.head())
test = pd.read_csv("nsmc_test.csv")

"""
print(train.shape)
print(test.shape)
print(train.head(10))
"""

sentences = train['review']
#print(sentences[:10])

sentences = ["[CLS] " + str(sentence) + " [SEP]" for sentence in sentences]
#print(sentences[:10])



labels = train['rating'].values
#print(labels)


tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

#print (sentences[0])
#print (tokenized_texts[0])


MAX_LEN = 128

# 토큰을 숫자 인덱스로 변환
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

# 문장을 MAX_LEN 길이에 맞게 자르고, 모자란 부분을 패딩 0으로 채움
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

input_ids[0]


attention_masks = []

# 어텐션 마스크를 패딩이 아니면 1, 패딩이면 0으로 설정
# 패딩 부분은 BERT 모델에서 어텐션을 수행하지 않아 속도 향상
for seq in input_ids:
    seq_mask = [float(i>0) for i in seq]
    attention_masks.append(seq_mask)

#print(attention_masks[0])


# 훈련셋과 검증셋으로 분리
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids,
                                                                                    labels,
                                                                                    random_state=2018,
                                                                                    test_size=0.1)

# 어텐션 마스크를 훈련셋과 검증셋으로 분리
train_masks, validation_masks, _, _ = train_test_split(attention_masks,
                                                       input_ids,
                                                       random_state=2018,
                                                       test_size=0.1)

# 데이터를 파이토치의 텐서로 변환
train_inputs = torch.tensor(train_inputs)
train_labels = torch.tensor(train_labels)
train_masks = torch.tensor(train_masks)
validation_inputs = torch.tensor(validation_inputs)
validation_labels = torch.tensor(validation_labels)
validation_masks = torch.tensor(validation_masks)

"""
print(train_inputs[0])
print(train_labels[0])
print(train_masks[0])
print(validation_inputs[0])
print(validation_labels[0])
print(validation_masks[0])
"""


# 배치 사이즈
batch_size = 32

# 파이토치의 DataLoader로 입력, 마스크, 라벨을 묶어 데이터 설정
# 학습시 배치 사이즈 만큼 데이터를 가져옴
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)



"""
# 리뷰 문장 추출
sentences = test['review']
sentences[:10]



# BERT의 입력 형식에 맞게 변환
sentences = ["[CLS] " + str(sentence) + " [SEP]" for sentence in sentences]
sentences[:10]


# 라벨 추출
labels = test['label'].values#
#labels
"""
# BERT의 토크나이저로 문장을 토큰으로 분리
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

print (sentences[0])
print (tokenized_texts[0])

# 입력 토큰의 최대 시퀀스 길이
MAX_LEN = 128

# 토큰을 숫자 인덱스로 변환
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

# 문장을 MAX_LEN 길이에 맞게 자르고, 모자란 부분을 패딩 0으로 채움
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

input_ids[0]


# 어텐션 마스크 초기화
attention_masks = []

# 어텐션 마스크를 패딩이 아니면 1, 패딩이면 0으로 설정
# 패딩 부분은 BERT 모델에서 어텐션을 수행하지 않아 속도 향상
for seq in input_ids:
    seq_mask = [float(i>0) for i in seq]
    attention_masks.append(seq_mask)

print(attention_masks[0])

# 데이터를 파이토치의 텐서로 변환
test_inputs = torch.tensor(input_ids)
test_labels = torch.tensor(labels)
test_masks = torch.tensor(attention_masks)

print(test_inputs[0])
print(test_labels[0])
print(test_masks[0])

# 배치 사이즈
batch_size = 32

# 파이토치의 DataLoader로 입력, 마스크, 라벨을 묶어 데이터 설정
# 학습시 배치 사이즈 만큼 데이터를 가져옴
test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)


# GPU 디바이스 이름 구함
device_name = tf.test.gpu_device_name()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 첫 번째 GPU 사용
# 디바이스 설정

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')


# 분류를 위한 BERT 모델 생성
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=4)
model.cuda()


# 옵티마이저 설정
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # 학습률
                  eps = 1e-8 # 0으로 나누는 것을 방지하기 위한 epsilon 값
                )

# 에폭수
epochs = 50               #**********************************************************************************

# 총 훈련 스텝 : 배치반복 횟수 * 에폭
total_steps = len(train_dataloader) * epochs

# 처음에 학습률을 조금씩 변화시키는 스케줄러 생성
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)


# 정확도 계산 함수
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# 시간 표시 함수
def format_time(elapsed):
    # 반올림
    elapsed_rounded = int(round((elapsed)))

    # hh:mm:ss으로 형태 변경
    return str(datetime.timedelta(seconds=elapsed_rounded))


import torch


# 모델 저장 함수
def save_model(model, epoch, filename='model.bin'):
    print(f"Saving model at epoch {epoch}...")
    torch.save(model.state_dict(), filename)
    print(f"Model saved as {filename}")


# 재현을 위해 랜덤시드 고정
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# 그래디언트 초기화
model.zero_grad()

# 에폭만큼 반복
for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # 시작 시간 설정
    t0 = time.time()

    # 로스 초기화
    total_loss = 0

    # 훈련모드로 변경
    model.train()

    # 데이터로더에서 배치만큼 반복하여 가져옴
    for step, batch in enumerate(train_dataloader):
        # 경과 정보 표시
        if step % 500 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # 배치를 GPU에 넣음
        batch = tuple(t.to(device) for t in batch)

        # 배치에서 데이터 추출
        b_input_ids, b_input_mask, b_labels = batch

        # Forward 수행
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels)

        # 로스 구함
        loss = outputs[0]

        # 총 로스 계산
        total_loss += loss.item()

        # Backward 수행으로 그래디언트 계산
        loss.backward()

        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 그래디언트를 통해 가중치 파라미터 업데이트
        optimizer.step()

        # 스케줄러로 학습률 감소
        scheduler.step()

        # 그래디언트 초기화
        model.zero_grad()

    # 평균 로스 계산
    avg_train_loss = total_loss / len(train_dataloader)

    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

    # 모델 저장 (매 에폭 후 저장)
    save_model(model, epoch_i + 1, f'model_epoch_{epoch_i + 1}.bin')

    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")

    # 시작 시간 설정
    t0 = time.time()

    # 평가모드로 변경
    model.eval()

    # 변수 초기화
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # 데이터로더에서 배치만큼 반복하여 가져옴
    for batch in validation_dataloader:
        # 배치를 GPU에 넣음
        batch = tuple(t.to(device) for t in batch)

        # 배치에서 데이터 추출
        b_input_ids, b_input_mask, b_labels = batch

        # 그래디언트 계산 안함
        with torch.no_grad():
            # Forward 수행
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)

        # 출력 로짓 구함
        logits = outputs[0]

        # CPU로 데이터 이동
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # 출력 로짓과 라벨을 비교하여 정확도 계산
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))

# 최종 모델 저장
save_model(model, epochs, 'final_model.bin')

print("")
print("Training complete!")

# 시작 시간 설정
t0 = time.time()

# 평가모드로 변경
model.eval()

# 변수 초기화
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0

# 데이터로더에서 배치만큼 반복하여 가져옴
for step, batch in enumerate(test_dataloader):
    # 경과 정보 표시
    if step % 100 == 0 and not step == 0:
        elapsed = format_time(time.time() - t0)
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(test_dataloader), elapsed))

    # 배치를 GPU에 넣음
    batch = tuple(t.to(device) for t in batch)

    # 배치에서 데이터 추출
    b_input_ids, b_input_mask, b_labels = batch

    # 그래디언트 계산 안함
    with torch.no_grad():
        # Forward 수행
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask)

    # 출력 로짓 구함
    logits = outputs[0]

    # CPU로 데이터 이동
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    # 출력 로짓과 라벨을 비교하여 정확도 계산
    tmp_eval_accuracy = flat_accuracy(logits, label_ids)
    eval_accuracy += tmp_eval_accuracy
    nb_eval_steps += 1

print("")
print("Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
print("Test took: {:}".format(format_time(time.time() - t0)))


# 입력 데이터 변환
def convert_input_data(sentences):
    # BERT의 토크나이저로 문장을 토큰으로 분리
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

    # 입력 토큰의 최대 시퀀스 길이
    MAX_LEN = 128

    # 토큰을 숫자 인덱스로 변환
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

    # 문장을 MAX_LEN 길이에 맞게 자르고, 모자란 부분을 패딩 0으로 채움
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # 어텐션 마스크 초기화
    attention_masks = []

    # 어텐션 마스크를 패딩이 아니면 1, 패딩이면 0으로 설정
    # 패딩 부분은 BERT 모델에서 어텐션을 수행하지 않아 속도 향상
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    # 데이터를 파이토치의 텐서로 변환
    inputs = torch.tensor(input_ids)
    masks = torch.tensor(attention_masks)

    return inputs, masks


# 문장 테스트
# 긍정으로 예측된 영화 제목을 저장할 리스트
positive_titles = ['Avatar']
funny_titles = []  # 웃음(2) 영화 리스트
scary_titles = []  # 공포(3) 영화 리스트

# 제목과 문장을 받아 4가지 레이블로 분류하는 함수
def test_sentences_with_titles_v2(titles, sentences):
    global positive_titles, funny_titles, scary_titles  # 전역 리스트 사용

    # 평가모드로 변경
    model.eval()

    # 문장만 입력 데이터로 변환
    inputs, masks = convert_input_data(sentences)

    # 데이터를 GPU에 넣음
    b_input_ids = inputs.to(device)
    b_input_mask = masks.to(device)

    # 그래디언트 계산 안함
    with torch.no_grad():
        # Forward 수행
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask)

    # 출력 로짓 구함
    logits = outputs[0]

    # CPU로 데이터 이동
    logits = logits.detach().cpu().numpy()

    # 예측 결과 저장
    predictions = np.argmax(logits, axis=1)
    for i, prediction in enumerate(predictions):
        if prediction == 1:  # 긍정으로 예측된 경우
            positive_titles.append(titles[i])
        elif prediction == 2:  # 웃음으로 예측된 경우
            funny_titles.append(titles[i])
        elif prediction == 3:  # 공포로 예측된 경우
            scary_titles.append(titles[i])

    return logits, predictions


logits, predictions = test_sentences_with_titles_v2(
    ['Titanic', 'Iron Man', 'Scary Movie', 'The Hangover'],
    ['감동적이고 아름다운 영화입니다.', '지루했지만 괜찮았어요.', '웃기면서도 무서운 영화였습니다.', '진짜 웃겨요, 대박!']
)

print("예측된 로짓:", logits)
print("예측된 레이블:", predictions)
print("긍정으로 예측된 영화 제목들:", positive_titles)
print("웃음으로 예측된 영화 제목들:", funny_titles)
print("공포로 예측된 영화 제목들:", scary_titles)




###################################################
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch

# 영화 데이터 로드
movies = pd.read_csv('tmdb_5000_movies.csv')






# BERT 모델과 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# 영화 개요(overview)를 BERT 임베딩으로 변환하는 함수
def embed_text(text):
    # 입력 데이터 토큰화
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 모델 추론
    with torch.no_grad():
        outputs = model(**inputs)

    # last_hidden_state를 평균내어 임베딩 반환
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()


# 전체 영화 개요 임베딩 생성
all_titles = movies['title'].tolist()
all_overviews = movies['overview'].fillna("").tolist()
all_embeddings = [embed_text(overview) for overview in all_overviews]

# 긍정 영화 리스트에 있는 영화들과 유사한 영화를 추천하는 함수
# 추천 함수 수정
def recommend_similar_movies_v2(selected_titles, all_titles, all_embeddings, top_k=5):
    recommendations = []

    # 선택된 영화 리스트에 있는 영화들의 임베딩을 가져옴
    selected_embeddings = [all_embeddings[all_titles.index(title)] for title in selected_titles if title in all_titles]

    # 선택된 영화들의 평균 임베딩 계산
    if selected_embeddings:
        mean_embedding = np.mean(selected_embeddings, axis=0).reshape(1, -1)
    else:
        print("선택된 영화 리스트가 비어있습니다.")
        return []

    # 모든 영화와의 코사인 유사도 계산
    all_embeddings = np.vstack(all_embeddings)
    similarities = cosine_similarity(mean_embedding, all_embeddings).flatten()

    # 유사도가 높은 순으로 영화 인덱스 정렬
    sorted_indices = similarities.argsort()[::-1]

    # 상위 추천 영화 선택 (이미 선택 리스트에 있는 영화는 제외)
    for idx in sorted_indices:
        title = all_titles[idx]
        if title not in selected_titles:
            recommendations.append(title)
            if len(recommendations) >= top_k:
                break

    return recommendations

# 웃음과 공포 영화 추천
recommended_funny_movies = recommend_similar_movies_v2(funny_titles, all_titles, all_embeddings, top_k=5)
recommended_scary_movies = recommend_similar_movies_v2(scary_titles, all_titles, all_embeddings, top_k=5)

print("웃음으로 예측된 영화 기반 추천:", recommended_funny_movies)
print("공포로 예측된 영화 기반 추천:", recommended_scary_movies)
