import streamlit as st
from PIL import Image
import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from groq import Groq
from gtts import gTTS
import pygame
from io import BytesIO
from functools import lru_cache
import re

# Initialize Groq and pygame
groq_api_key = "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"
client = Groq(api_key=groq_api_key)

if not pygame.mixer.get_init():
    pygame.mixer.init()

# Language support dictionary
supported_languages = {
    "English": {
        "code": "en", 
        "tts_code": "en",
        "translations": {
            "title": "Plant Disease Detection",
            "upload": "Upload an image",
            "prediction": "Predicted Disease",
            "confidence": "Confidence",
            "treatment": "Recommended Treatment",
            "uploaded_image": "Uploaded Image",
            "footer": "© 2024 Plant Disease Detection System"
        }
    },
    "ಕನ್ನಡ (Kannada)": {
        "code": "kn", 
        "tts_code": "kn",
        "translations": {
            "title": "ಸಸ್ಯ ರೋಗ ಪತ್ತೆ",
            "upload": "ಚಿತ್ರವನ್ನು ಅಪ್ಲೋಡ್ ಮಾಡಿ",
            "prediction": "ಊಹಿಸಿದ ರೋಗ",
            "confidence": "ವಿಶ್ವಾಸ",
            "treatment": "ಶಿಫಾರಸು ಮಾಡಿದ ಚಿಕಿತ್ಸೆ",
            "uploaded_image": "ಅಪ್ಲೋಡ್ ಮಾಡಿದ ಚಿತ್ರ",
            "footer": "© 2024 ಸಸ್ಯ ರೋಗ ಪತ್ತೆ ವ್ಯವಸ್ಥೆ"
        }
    },
    "తెలుగు (Telugu)": {
        "code": "te", 
        "tts_code": "te",
        "translations": {
            "title": "మొక్క వ్యాధి గుర్తింపు",
            "upload": "చిత్రాన్ని అప్లోడ్ చేయండి",
            "prediction": "అంచనా వేసిన వ్యాధి",
            "confidence": "నమ్మకం",
            "treatment": "సిఫార్సు చేసిన చికిత్స",
            "uploaded_image": "అప్లోడ్ చేసిన చిత్రం",
            "footer": "© 2024 మొక్క వ్యాధి గుర్తింపు వ్యవస్థ"
        }
    },
    "தமிழ் (Tamil)": {
        "code": "ta", 
        "tts_code": "ta",
        "translations": {
            "title": "தாவர நோய் கண்டறிதல்",
            "upload": "படத்தை பதிவேற்றவும்",
            "prediction": "கணித்த நோய்",
            "confidence": "நம்பிக்கை",
            "treatment": "பரிந்துரைக்கப்பட்ட சிகிச்சை",
            "uploaded_image": "பதிவேற்றிய படம்",
            "footer": "© 2024 தாவர நோய் கண்டறிதல் அமைப்பு"
        }
    },
    "മലയാളം (Malayalam)": {
        "code": "ml", 
        "tts_code": "ml",
        "translations": {
            "title": "സസ്യ രോഗ നിർണ്ണയം",
            "upload": "ചിത്രം അപ്‌ലോഡ് ചെയ്യുക",
            "prediction": "പ്രവചിച്ച രോഗം",
            "confidence": "ആത്മവിശ്വാസം",
            "treatment": "ശുപാർശ ചെയ്ത ചികിത്സ",
            "uploaded_image": "അപ്‌ലോഡ് ചെയ്ത ചിത്രം",
            "footer": "© 2024 സസ്യ രോഗ നിർണ്ണയ സംവിധാനം"
        }
    },
    "हिंदी (Hindi)": {
        "code": "hi", 
        "tts_code": "hi",
        "translations": {
            "title": "पौधों की बीमारी की पहचान",
            "upload": "छवि अपलोड करें",
            "prediction": "अनुमानित बीमारी",
            "confidence": "विश्वास",
            "treatment": "अनुशंसित उपचार",
            "uploaded_image": "अपलोड की गई छवि",
            "footer": "© 2024 पौधों की बीमारी पहचान प्रणाली"
        }
    }
}

# Disease treatments dictionary
disease_treatments = {
    "Grape with Black Rot": "Prune affected areas, avoid water on leaves, use fungicide if severe.",
    "Potato with Early Blight": "Apply fungicides, avoid overhead watering, rotate crops yearly.",
    "Tomato with Early Blight": "Remove infected leaves, use copper-based fungicide, maintain good airflow.",
    "Apple with Scab": "Remove fallen leaves, prune trees, apply fungicide in early spring.",
    "Wheat with Leaf Rust": "Apply resistant varieties, use fungicides, remove weeds.",
    "Cucumber with Downy Mildew": "Use resistant varieties, ensure good air circulation, apply fungicide.",
    "Rose with Powdery Mildew": "Use sulfur or potassium bicarbonate sprays, prune affected areas.",
    "Strawberry with Gray Mold": "Remove infected fruits, improve ventilation, avoid wetting fruit.",
    "Peach with Leaf Curl": "Apply fungicide in late fall or early spring, remove affected leaves.",
    "Banana with Panama Disease": "Use disease-resistant varieties, ensure soil drainage.",
    "Tomato with Septoria Leaf Spot": "Use resistant varieties, remove infected leaves, apply fungicide.",
    "Corn with Smut": "Remove infected ears, use disease-free seed, rotate crops.",
    "Carrot with Root Rot": "Ensure well-draining soil, avoid excessive watering.",
    "Onion with Downy Mildew": "Use fungicides, ensure adequate spacing.",
    "Potato with Late Blight": "Apply copper-based fungicides, remove affected foliage.",
    "Citrus with Greening Disease": "Remove infected trees, control leafhopper population.",
    "Lettuce with Downy Mildew": "Ensure good air circulation, avoid overhead watering.",
    "Pepper with Bacterial Spot": "Use resistant varieties, apply copper-based bactericides.",
    "Eggplant with Verticillium Wilt": "Use resistant varieties, solarize soil before planting.",
    "Cotton with Boll Rot": "Improve drainage, remove infected bolls, apply fungicides if necessary.",
    "Soybean with Soybean Rust": "Use fungicides, rotate crops, use resistant varieties if available.",
    "Rice with Sheath Blight": "Reduce nitrogen application, maintain proper water levels.",
    "Sunflower with Downy Mildew": "Use resistant varieties, avoid waterlogging.",
    "Barley with Net Blotch": "Use resistant varieties, remove crop residues.",
    "Oat with Crown Rust": "Use resistant varieties, apply fungicides.",
    "Sugarcane with Red Rot": "Use disease-free cuttings, control weeds.",
    "Pine with Pine Wilt": "Remove and destroy infected trees, control beetle population.",
    "Avocado with Anthracnose": "Prune infected branches, use copper-based fungicides.",
    "Papaya with Papaya Ringspot Virus": "Use virus-resistant varieties, remove infected plants.",
    "Mango with Powdery Mildew": "Use sulfur-based fungicides, remove affected parts.",
    "Peanut with Leaf Spot": "Use resistant varieties, apply fungicides, rotate crops.",
    "Chili with Anthracnose": "Apply copper fungicides, remove infected fruits.",
    "Strawberry with Leaf Scorch": "Remove infected leaves, maintain proper plant spacing, use fungicides containing captan or copper, ensure good air circulation, avoid overhead watering, mulch to prevent soil splash.",
    "Garlic with White Rot": "Remove infected plants, improve soil drainage."
}

@lru_cache(maxsize=1)
def load_model_and_extractor():
    extractor = AutoFeatureExtractor.from_pretrained("linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification")
    model = AutoModelForImageClassification.from_pretrained("linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification")
    return extractor, model

def get_translation(key, language):
    return supported_languages[language]["translations"][key]

def sanitize_text(text):
    sanitized = re.sub(r'[\*\.,#]', '', text)
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    return sanitized

def format_confidence(value):
    return f"{value:.2f} percent"

import re

def clean_text_for_speech(text):
    # Remove emojis
    emoji_pattern = re.compile("[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\u2702-\u27B0\u24C2-\U0001F251]+", flags=re.UNICODE)
    text = emoji_pattern.sub('', text)
    
    # Format percentages
    percentage_pattern = re.compile(r'(\d+\.\d+)%')
    text = percentage_pattern.sub(lambda m: f"{float(m.group(1))} percent", text)
    
    # Remove markdown formatting including asterisks
    text = re.sub(r'\*\*|\*', '', text)  # Remove both single and double asterisks
    text = re.sub(r'#+\s?', '', text)    # Remove markdown headers
    text = re.sub(r'\[(.*?)\]', r'\1', text)
    
    # Remove multiple spaces and newlines
    text = ' '.join(text.split())
    
    return text

def text_to_speech(text, language):
    try:
        cleaned_text = clean_text_for_speech(text)
        tts = gTTS(
            text=cleaned_text,
            lang=supported_languages[language]["tts_code"],
            slow=False
        )
        audio_bytes = BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except Exception as e:
        st.error(f"TTS Error: {str(e)}")
        return None

def toggle_audio():
    if st.session_state.is_playing:
        pygame.mixer.music.pause()
    else:
        if not pygame.mixer.music.get_busy():
            audio_bytes = text_to_speech(st.session_state.current_text, st.session_state.selected_language)
            if audio_bytes:
                pygame.mixer.music.load(audio_bytes, 'mp3')
                pygame.mixer.music.play()
        else:
            pygame.mixer.music.unpause()
    st.session_state.is_playing = not st.session_state.is_playing

def generate_groq_report(disease, confidence, symptoms, lang):
    system_prompt = f"""You are a plant disease expert. Provide analysis in {lang}.
    Generate a single detailed report covering disease details, symptoms, treatment, and prevention.
    Keep the response concise and well-formatted with emojis."""

    user_prompt = f"""
    Plant Disease Analysis:
    - Disease: {disease}
    - Confidence: {confidence:.2f}%
    - Symptoms: {symptoms}
    
    Provide ONE comprehensive report in {lang}."""

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama3-70b-8192",
            temperature=0.7,
            max_tokens=1000  # Reduced to prevent duplicate content
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating report: {str(e)}"

# Initialize session states
if 'selected_language' not in st.session_state:
    st.session_state.selected_language = "English"
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False
if 'current_text' not in st.session_state:
    st.session_state.current_text = ""

# Main UI
selected_language = st.selectbox(
    "Select Language", 
    list(supported_languages.keys()), 
    index=list(supported_languages.keys()).index(st.session_state.selected_language)
)
st.session_state.selected_language = selected_language

st.title(f"🌿 {get_translation('title', selected_language)} 🌿")

uploaded_file = st.file_uploader(
    get_translation('upload', selected_language), 
    type=["jpg", "jpeg", "png", "bmp"]
)  # Added missing closing parenthesis

# Update report generation section
if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption=get_translation('uploaded_image', selected_language))
    
    # Process image
    extractor, model = load_model_and_extractor()
    inputs = extractor(image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()
    confidence = torch.nn.functional.softmax(logits, dim=-1)[0][predicted_class].item()
    
    # Get prediction and format confidence
    predicted_disease = model.config.id2label[predicted_class]
    formatted_confidence = f"{confidence * 100:.2f}%"
    
    # Generate and store report with confidence
    report_content = f"""# 🔍 Disease Detection Results

**Accuracy**: {formatted_confidence}

{generate_groq_report(
    predicted_disease,
    confidence * 100,
    "Visible leaf damage and discoloration",
    selected_language
)}"""
    
    st.session_state.report_content = report_content
    st.session_state.current_text = st.session_state.report_content
    
    # Display report with columns
    col1, col2 = st.columns([0.9, 0.1])
    with col1:
        st.markdown(st.session_state.report_content)
    with col2:
        if st.button("🔊" if not st.session_state.is_playing else "🔈", 
                    key="audio_button"):
            audio_bytes = text_to_speech(st.session_state.report_content, selected_language)
            if audio_bytes:
                st.audio(audio_bytes, format='audio/mp3')

# Add footer
st.markdown("---")
st.markdown(get_translation('footer', selected_language))