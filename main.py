import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import ElectraForSequenceClassification, ElectraTokenizer, AdamW, get_cosine_schedule_with_warmup
from dataset import TextDataset

# 데이터 읽기
data = pd.read_csv("your_data.csv")
train_data, val_data = train_test_split(data, test_size=0.1)

# 손실 함수
def model_loss(model, inputs, return_outputs=False):
    labels = inputs.pop('label')
    outputs = model(**inputs)
    logits = outputs.logits
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(logits, labels)
    return (loss, outputs) if return_outputs else loss

# 모델과 토크나이저 생성
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

model_sbj = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator", num_labels = 3)

train_dataset_sbj = TextDataset(train_data, tokenizer, label_idx=1)
val_dataset_sbj = TextDataset(val_data, tokenizer, label_idx=1)

train_loader_sbj = DataLoader(train_dataset_sbj, batch_size=16, shuffle=True)
val_loader_sbj = DataLoader(val_dataset_sbj, batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_sbj = model_sbj.to(device)

optimizer_sbj = AdamW(model_sbj.parameters(), lr = 5e-5)

NUM_EPOCHS = 5
total_steps = len(train_loader_sbj) * NUM_EPOCHS
scheduler_sbj = get_cosine_schedule_with_warmup(optimizer_sbj, num_warmup_steps = int(total_steps * 0.1), num_training_steps = total_steps)

# Training and Evaluation
for epoch in range(NUM_EPOCHS):
    # Model 1 Training
    model_sbj.train()
    for batch in train_loader_sbj:
        optimizer_sbj.zero_grad()
        inputs = {key: val.to(device) for key, val in batch.items()}
        loss = model_loss(model_sbj, inputs)
        loss.backward()
        optimizer_sbj.step()
        scheduler_sbj.step()
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

    # Model 1 Evaluation
    model_sbj.eval()
    total_eval_accuracy_sbj = 0
    for batch in val_loader_sbj:
        inputs = {key: val.to(device) for key, val in batch.items()}
        with torch.no_grad():
            outputs = model_sbj(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            label_ids = inputs['label']
            total_eval_accuracy_sbj += (preds == label_ids).sum().item()
    print(f"Model_sbj Epoch {epoch}: Validation Accuracy: {total_eval_accuracy_sbj / len(val_dataset_sbj)}")
    
model_sbj.save_pretrained("saved_models/model_sbj")