import streamlit as st
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
import json

# Load model & tokenizer
@st.cache_resource
def load_model():
    model_path = "Syetsuki/hoax-detector"
    tokenizer = BertTokenizerFast.from_pretrained(f"{model_path}/tokenizer")
    model = BertForSequenceClassification.from_pretrained(f"{model_path}")
    model.eval()
    return tokenizer, model


tokenizer, model = load_model()

id2label = {
    0: "Non-Hoax",
    1: "Hoax"
}

def predict_news(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
    probs = probs[0]
    pred_id = torch.argmax(probs).item()
    confidence = probs[pred_id].item()
    result = {
        'prediction': id2label[pred_id].upper(),
        'confidence': f"{confidence * 100:.1f}%",
        'probabilities': {
            'Non-Hoax': f"{probs[0].item():.3f}",
            'Hoax': f"{probs[1].item():.3f}"
        }
    }
    return result

# Streamlit UI
st.title("📰 Hoax Detector IndoBERT")
text_input = st.text_area("Masukkan judul atau isi berita:")

if st.button("Deteksi"):
    if text_input.strip() == "":
        st.warning("Tolong masukkan teks terlebih dahulu.")
    else:
        result = predict_news(text_input)
        st.success("Hasil Deteksi:")
        st.json(result)
