import torch
from transformers import BertTokenizer, BertForSequenceClassification, ElectraTokenizer, ElectraForSequenceClassification

def classify_text(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
    return preds.item()

test_bool = True

while test_bool:
    text = input("텍스트를 입력하세요: ")
    if text != 'q':
        # 모델 및 토크나이저 로드
        tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
        model = ElectraForSequenceClassification.from_pretrained("saved_models/koelectra_model")
        # tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
        # model = BertForSequenceClassification.from_pretrained("saved_models/kobert_model")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        result = classify_text(text, tokenizer, model, device)
        if result == 0:
            print(f'{result}, 휴/복학')
        elif result == 1:
            print(f'{result}, 졸업')
        else:
            print(f'{result}, 수강')
    else:
        test_bool = False
    