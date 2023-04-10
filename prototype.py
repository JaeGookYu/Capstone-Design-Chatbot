import torch
from transformers import ElectraTokenizer, ElectraForSequenceClassification

def predict(model, tokenizer, device, input_text):
    encoded_input = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    for key, val in encoded_input.items():
        encoded_input[key] = val.to(device)

    model.eval()

    with torch.no_grad():
        outputs = model(**encoded_input)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        predicted_label = preds.item()

    return predicted_label

if __name__ == "__main__":
    
    input_text = input("문장 입력:")
    
    # 토크나이저와 모델 로드
    tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    prt_model = ElectraForSequenceClassification.from_pretrained("path/to/save/model_sbj")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prt_model = prt_model.to(device)

    predicted_label = predict(prt_model, tokenizer, device, input_text)
    
    print(f"Input: {input_text}")
    print(f"Predicted Label: {predicted_label}")