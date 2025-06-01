import streamlit as st
import torch
import os
from transformers import BertTokenizerFast, BertForSequenceClassification

# Ganti dengan username dan nama repository Hugging Face kamu
HF_USERNAME = "Syetsuki"  # ← GANTI INI
HF_REPO_NAME = "hoax_detector"  # ← PASTIKAN NAMA REPO BENAR

@st.cache_resource
def load_model():
    # Login ke Hugging Face jika ada token (untuk private repos)
    try:
        from huggingface_hub import login
        if "HF_TOKEN" in os.environ:
            login(token=os.environ["HF_TOKEN"])
    except ImportError:
        pass  # huggingface_hub mungkin tidak diinstall
    
    try:
        model_name = f"{HF_USERNAME}/{HF_REPO_NAME}"
        
        # Load tokenizer dari Hugging Face
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
        
        # Load model dari Hugging Face
        model = BertForSequenceClassification.from_pretrained(model_name)
        
        st.success("✅ Model berhasil dimuat!")
        return tokenizer, model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("💡 Pastikan:")
        st.info("1. Username dan nama repository Hugging Face benar")
        st.info("2. Repository bersifat public atau token HF sudah diset")
        st.info("3. Files model lengkap di repository")
        return None, None

def predict_hoax(text, tokenizer, model):
    # Preprocessing
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Prediction
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions.max().item()
    
    # Assuming 0 = Not Hoax, 1 = Hoax
    label = "Hoax" if predicted_class == 1 else "Not Hoax"
    
    return label, confidence

def main():
    st.title("🔍 Hoax Detector")
    st.write("Masukkan teks berita untuk mendeteksi apakah itu hoax atau bukan.")
    
    # Load model
    tokenizer, model = load_model()
    
    if tokenizer is None or model is None:
        st.error("Gagal memuat model. Silakan coba lagi nanti.")
        return
    
    # Input text
    text_input = st.text_area(
        "Masukkan teks berita:",
        height=200,
        placeholder="Ketik atau paste teks berita di sini..."
    )
    
    if st.button("🔍 Analisis"):
        if text_input.strip():
            with st.spinner("Menganalisis..."):
                try:
                    label, confidence = predict_hoax(text_input, tokenizer, model)
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if label == "Hoax":
                            st.error(f"⚠️ **{label}**")
                        else:
                            st.success(f"✅ **{label}**")
                    
                    with col2:
                        st.metric("Confidence", f"{confidence:.2%}")
                    
                    # Progress bar for confidence
                    st.progress(confidence)
                    
                except Exception as e:
                    st.error(f"Error dalam prediksi: {str(e)}")
        else:
            st.warning("Silakan masukkan teks untuk dianalisis.")

if __name__ == "__main__":
    main()
