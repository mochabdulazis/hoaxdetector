import streamlit as st

# Import dengan error handling untuk PyTorch
try:
    import torch
    torch.set_num_threads(1)  # Mengurangi konflik threading
    from transformers import BertTokenizerFast, BertForSequenceClassification
except Exception as e:
    st.error(f"Error importing libraries: {e}")
    st.stop()

# Repository info - PUBLIC, tidak butuh token
MODEL_NAME = "syetsuki/hoax-detector"

@st.cache_resource
def load_model():
    try:
        st.info(f"Loading model: {MODEL_NAME}")
        
        # Load langsung tanpa login (karena public repo)
        tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
        model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
        
        return tokenizer, model
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
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
    
    label = "Hoax" if predicted_class == 1 else "Not Hoax"
    return label, confidence

def main():
    st.title("🔍 Hoax Detector")
    st.write("Deteksi berita hoax menggunakan AI")
    
    # Load model
    tokenizer, model = load_model()
    
    if tokenizer is None or model is None:
        st.error("Gagal memuat model")
        return
    
    st.success("✅ Model siap digunakan!")
    
    # Input
    text_input = st.text_area(
        "Masukkan teks berita:",
        height=150,
        placeholder="Paste teks berita di sini..."
    )
    
    if st.button("🔍 Analisis"):
        if text_input.strip():
            with st.spinner("Menganalisis..."):
                try:
                    label, confidence = predict_hoax(text_input, tokenizer, model)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if label == "Hoax":
                            st.error(f"⚠️ **{label}**")
                        else:
                            st.success(f"✅ **{label}**")
                    
                    with col2:
                        st.metric("Confidence", f"{confidence:.1%}")
                    
                    st.progress(confidence)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Masukkan teks terlebih dahulu")

if __name__ == "__main__":
    main()
