import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from dataset import KoBERTDataset

# 하이퍼파라미터 설정
batch_size = 32
max_length = 128
learning_rate = 5e-5
epochs = 4

# 토크나이저 및 모델 초기화
tokenizer = BertTokenizer.from_pretrained('monologg/kobert')
model = BertForSequenceClassification.from_pretrained('monologg/kobert', num_labels = 2)  # num_labels에 라벨 수 입력
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 데이터 로드
train_dataset = KoBERTDataset('train.csv', tokenizer, max_length)
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

# 옵티마이저 및 스케줄러 설정
optimizer = AdamW(model.parameters(), lr = learning_rate)
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

# 모델 학습
model.train()
for epoch in range(epochs):
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")