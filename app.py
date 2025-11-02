import streamlit as st
from PIL import Image
import os
import io
import time

# Import the core detection logic, which also initializes the database
from model import detect_emotion, EMOTION_EMOJI, get_all_emotions

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Facial Emotion Detection",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- CACHING AND UTILITIES ---

# To avoid re-running the main logic on every interaction
@st.cache_data
def load_emotions():
    return get_all_emotions()

# --- STREAMLIT APP LAYOUT ---

def main():
    st.title(f"Emotion Detection Web App {EMOTION_EMOJI.get('happy', '')}")
    st.markdown("---")

    st.header("Upload an Image")
    uploaded_file = st.file_uploader(
        "Choose a high-quality image of a face...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Create a temporary file path to pass to DeepFace
        temp_file_path = f"temp_{int(time.time())}_{uploaded_file.name}"
        
        # 1. Save the uploaded file to disk
        try:
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 2. Display the uploaded image
            st.image(temp_file_path, caption='Uploaded Image', use_column_width=True)
            
            # 3. Process the image
            with st.spinner('Analyzing emotion...'):
                dominant_emotion, confidence = detect_emotion(temp_file_path)
            
            st.success("Analysis Complete!")
            
            # 4. Display results
            if dominant_emotion != "Error":
                emoji = EMOTION_EMOJI.get(dominant_emotion, '❓')
                st.subheader(f"Result: {emoji} {dominant_emotion.upper()}")
                st.info(f"Confidence: **{confidence:.2f}%**")
                
            else:
                st.error("Could not process the image. Ensure a face is clearly visible.")

        except Exception as e:
            st.error(f"An unexpected error occurred during processing: {e}")
            
        finally:
            # 5. Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    else:
        st.info("Upload an image to start the facial emotion detection.")
    
    st.markdown("---")
    st.caption("Project built using DeepFace and Streamlit, satisfying the assignment requirements.")

if __name__ == "__main__":
    main()