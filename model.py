from emotiefflib import EmotiEff
import cv2
import numpy as np

# Initialize the model
model = EmotiEff()

# Define supported emotions (you can adjust this depending on what the library supports)
EMOTION_EMOJI = {
    "happy": "😄",
    "sad": "😢",
    "angry": "😠",
    "surprise": "😲",
    "neutral": "😐",
    "fear": "😨",
    "disgust": "🤢"
}

def get_all_emotions():
    """Return a list of supported emotions."""
    return list(EMOTION_EMOJI.keys())

def detect_emotion(img):
    """
    Detect emotion from an image (numpy array or path).
    Returns (emotion, confidence)
    """
    try:
        # Ensure the image is in RGB format
        if isinstance(img, np.ndarray):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Expected numpy array for image input.")

        # Run prediction using EmotiEff
        result = model.predict(img_rgb)

        # Extract emotion and confidence
        emotion = result["emotion"]
        confidence = result["confidence"]

        return emotion.lower(), confidence

    except Exception as e:
        print("Error detecting emotion:", e)
        return "Error", 0.0
