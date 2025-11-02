import streamlit as st
import cv2
import numpy as np
from PIL import Image
from model import detect_emotion, get_all_emotions, EMOTION_EMOJI

st.set_page_config(page_title="Emotion Detection Web App", page_icon="😊", layout="centered")

st.title("🎭 Emotion Detection System")
st.markdown(
    """
    Upload an image and the system will predict the **dominant emotion** displayed.
    This version uses **EmotiEffLib (v1.1.1)** for fast and lightweight emotion recognition.
    """
)

# Sidebar info
st.sidebar.title("About")
st.sidebar.info(
    """
    **Developer:** Olubadejo Folajuwon  
    **Library:** EmotiEffLib v1.1.1  
    **Framework:** Streamlit  
    """
)

# File uploader
uploaded_file = st.file_uploader("📸 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Convert uploaded file to OpenCV image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    with st.spinner("Detecting emotion... ⏳"):
        emotion, confidence = detect_emotion(img_bgr)

    if emotion != "Error":
        st.success(f"**Emotion:** {emotion.capitalize()} {EMOTION_EMOJI.get(emotion, '')}")
        st.write(f"**Confidence:** {confidence:.2f}")
    else:
        st.error("Could not detect emotion. Please try another image.")

else:
    st.info("Please upload an image to begin.")

# Optional: show supported emotions
st.markdown("---")
st.subheader("Supported Emotions")
st.write(", ".join([f"{emo.capitalize()} {EMOTION_EMOJI[emo]}" for emo in get_all_emotions()]))
