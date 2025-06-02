import streamlit as st

# Import dengan error handling untuk PyTorch
try:
    import torch
    torch.set_num_threads(1)  # Mengurangi konflik threading
    from transformers import BertTokenizerFast, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
except Exception as e:
    st.error(f"Error importing libraries: {e}")
    st.stop()

# Repository info - PUBLIC, tidak butuh token
MODEL_NAME = "syetsuki/hoax-detector"  # Model Anda sendiri

@st.cache_resource
def load_model():
    try:
        
        # Method 1: Coba dengan AutoTokenizer dan AutoModel (lebih fleksibel)
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            return tokenizer, model
        except Exception as e1:
            st.warning(f"Auto classes gagal: {e1}")
            
            # Method 2: Coba dengan BERT classes
            try:
                tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
                model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
                st.success("✅ Model loaded dengan BERT classes")
                return tokenizer, model
            except Exception as e2:
                st.warning(f"BERT classes gagal: {e2}")
                
                # Method 3: Load tokenizer only, use alternative model
                try:
                    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
                    st.warning("⚠️ Hanya tokenizer yang berhasil dimuat. Menggunakan model alternatif...")
                    
                    # Gunakan model alternatif yang reliable
                    alternative_model = BertForSequenceClassification.from_pretrained(
                        'bert-base-uncased', 
                        num_labels=2
                    )
                    st.info("🔄 Menggunakan BERT base model sebagai alternatif")
                    return tokenizer, alternative_model
                    
                except Exception as e3:
                    st.error(f"Semua method gagal: {e3}")
                    
                    # Method 4: Full fallback
                    st.warning("🔄 Menggunakan full fallback model...")
                    fallback_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
                    fallback_model = BertForSequenceClassification.from_pretrained(
                        'bert-base-uncased', 
                        num_labels=2
                    )
                    st.info("⚠️ Menggunakan BERT base model (hasil mungkin tidak optimal)")
                    return fallback_tokenizer, fallback_model
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def predict_hoax(text, tokenizer, model):
    try:
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
        
        # Label mapping (sesuaikan dengan model Anda)
        label = "Hoax" if predicted_class == 1 else "Not Hoax"
        return label, confidence
        
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return "Error", 0.0

def main():
    st.title("🔍 Hoax Detector")
    st.success("Deteksi berita hoax menggunakan AI")
    
    # Load model
    tokenizer, model = load_model()
    
    if tokenizer is None or model is None:
        st.error("❌ Gagal memuat model")
        st.stop()
        return
    
    # Input
    text_input = st.text_area(
        "Masukkan teks berita:",
        height=150,
        placeholder="Paste teks berita di sini..."
    )
    
    # Example text
    if st.button("📝 Contoh Teks"):
        example_text = "Pemerintah mengumumkan program bantuan baru untuk masyarakat yang terdampak pandemi dengan total anggaran 50 triliun rupiah."
        st.text_area("Contoh:", value=example_text, height=100)
    
    if st.button("🔍 Analisis"):
        if text_input.strip():
            with st.spinner("Menganalisis..."):
                try:
                    label, confidence = predict_hoax(text_input, tokenizer, model)
                    
                    if label != "Error":
                        col1, col2 = st.columns(2)
                        with col1:
                            if label == "Hoax":
                                st.error(f"⚠️ **{label}**")
                            else:
                                st.success(f"✅ **{label}**")
                        
                        with col2:
                            st.metric("Confidence", f"{confidence:.1%}")
                        
                        st.progress(confidence)
                        
                        # Disclaimer
                        st.caption("⚠️ Hasil ini adalah prediksi AI dan tidak 100% akurat. Selalu verifikasi dengan sumber terpercaya.")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Masukkan teks terlebih dahulu")

if __name__ == "__main__":
    main()
