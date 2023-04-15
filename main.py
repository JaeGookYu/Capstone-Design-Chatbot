import pandas as pd
import torch
import torch.nn as nn
from dataset import TextDataset
from torch.utils.data import DataLoader
from transformers import ElectraForSequenceClassification, ElectraTokenizer,BertForSequenceClassification,BertTokenizer, AdamW, get_cosine_schedule_with_warmup

# 데이터 읽기
data = pd.read_csv("data/data.csv",encoding='cp949')

# 데이터 분할 비율 설정 (예: 70% train, 30% test)
train_ratio = 0.7
test_ratio = 0.3
train_data = data.sample(frac=train_ratio)
val_data = data.drop(train_data.index)

# 모델과 토크나이저 생성
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator", num_labels=3)

# tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
# model = BertForSequenceClassification.from_pretrained("monologg/kobert", num_labels=3)

# 데이터셋과 데이터 로더 생성
train_dataset = TextDataset(train_data, tokenizer)
val_dataset = TextDataset(val_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 옵티마이저와 스케줄러 설정
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 5
total_steps = len(train_loader) * num_epochs
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
loss_fct = nn.CrossEntropyLoss()

# 학습 및 평가
for epoch in range(num_epochs):
    # 학습
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        inputs = {key: val.to(device) for key, val in batch.items()}
        labels = inputs.pop('label')
        outputs = model(**inputs)
        logits = outputs.logits
        loss = loss_fct(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

    # 평가
    model.eval()
    total_eval_accuracy = 0
    for batch in val_loader:
        inputs = {key: val.to(device) for key, val in batch.items()}
        labels = inputs.pop('label')
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            total_eval_accuracy += (preds == labels).sum().item()

    # 정확도 출력
    print(f"Epoch {epoch + 1}/{num_epochs}: Validation Accuracy: {total_eval_accuracy / len(val_dataset)}")
model.save_pretrained("saved_models/koelectra_model")
# model.save_pretrained("saved_models/kobert_model")
