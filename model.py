import numpy as np
import cv2
import sqlite3
from deepface import DeepFace

# --- CONFIGURATION ---

# 1. Map DeepFace emotions to an emoji for display
EMOTION_EMOJI = {
    'happy': '😊',
    'sad': '😞',
    'angry': '😡',
    'surprise': '😮',
    'fear': '😨',
    'disgust': '🤢',
    'neutral': '😐',
}

def get_all_emotions():
    """Returns a list of all supported emotion keys."""
    return list(EMOTION_EMOJI.keys())


# --- CORE LOGIC ---

def initialize_db():
    """Initializes the SQLite database and table."""
    conn = sqlite3.connect('emotion_detection.db')
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            emotion TEXT NOT NULL,
            confidence REAL NOT NULL,
            image_path TEXT,
            result_blob BLOB
        )
    """)
    conn.commit()
    conn.close()

def save_prediction(emotion, confidence, image_path=None, result_blob=None):
    """Saves the prediction result to the database."""
    conn = sqlite3.connect('emotion_detection.db')
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (emotion, confidence, image_path, result_blob) 
        VALUES (?, ?, ?, ?)
    """, (emotion, confidence, image_path, result_blob))
    conn.commit()
    conn.close()

def detect_emotion(img_path_or_array):
    """
    Detects the dominant emotion from an image using DeepFace.
    Returns the dominant emotion and its confidence score.
    """
    try:
        # DeepFace expects the image as a path or a NumPy array.
        # Since we handle file uploads, the input is already a file object/path.
        result = DeepFace.analyze(
            img_path_or_array, 
            actions=['emotion'], 
            enforce_detection=False
        )

        if not result:
            return "No Face Detected", 0.0

        # DeepFace returns a list of results, we take the first one
        dominant_emotion = result[0]['dominant_emotion']
        confidence = result[0]['emotion'][dominant_emotion]

        # Save result to the database
        save_prediction(dominant_emotion, confidence)

        return dominant_emotion, confidence

    except Exception as e:
        print(f"DeepFace Analysis Error: {e}")
        return "Error", 0.0

# Initialize DB when model.py is imported
initialize_db()