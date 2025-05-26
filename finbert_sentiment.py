from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# ----------------------------
# Load Original FinBERT
# ----------------------------
orig_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
orig_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
orig_model.eval()

# ----------------------------
# Load Fine-Tuned FinBERT
# ----------------------------
finetuned_path = "./finetuned-finbert"  # ✅ make sure folder is in the same directory
ft_tokenizer = AutoTokenizer.from_pretrained(finetuned_path)
ft_model = AutoModelForSequenceClassification.from_pretrained(finetuned_path)
ft_model.eval()

# ----------------------------
# Common Setup
# ----------------------------
label_to_score = {"positive": 1, "neutral": 0, "negative": -1}
id2label = {0: "negative", 1: "neutral", 2: "positive"}


# ----------------------------
# Base Prediction Logic
# ----------------------------
def predict_single(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
    label = id2label[pred_class]
    score = label_to_score[label]
    return label, confidence, score


# ----------------------------
# Compare Both Models on Each Text
# ----------------------------
def compare_sentiment_models(texts):
    if isinstance(texts, str):
        texts = [texts]

    results = []

    for text in texts:
        orig_label, orig_conf, orig_score = predict_single(text, orig_model, orig_tokenizer)
        ft_label, ft_conf, ft_score = predict_single(text, ft_model, ft_tokenizer)

        result = {
            "text": text,
            "original": {
                "label": orig_label,
                "confidence": round(orig_conf, 3),
                "score": orig_score
            },
            "finetuned": {
                "label": ft_label,
                "confidence": round(ft_conf, 3),
                "score": ft_score
            }
        }

        results.append(result)

        print(f"\n📝 Text: {text}")
        print(f"🔹 Original FinBERT:  {orig_label} ({orig_conf*100:.2f}%)")
        print(f"🔸 Fine-Tuned FinBERT: {ft_label} ({ft_conf*100:.2f}%)")

    return results


# ✅ Basic wrapper to return a single average score (used by reddit_sentiment.py)
def get_sentiment(texts, return_labels=False):
    result = compare_sentiment_models(texts)
    if return_labels:
        return [r["finetuned"]["label"] for r in result]
    else:
        scores = [r["finetuned"]["score"] for r in result]
        return round(sum(scores) / len(scores), 3) if scores else 0



