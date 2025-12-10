
import os
import shutil
import logging
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
import numpy as np

# Import our modules
from services.text_extraction import extract_text_from_file
from services.ats_score import calculate_ats_score
from services.resume_parser import parse_resume
from fastapi import BackgroundTasks

# Import Tensorflow/Keras for inference
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Resume Analyzer API", version="1.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model/tokenizer
model: Optional[Model] = None
tokenizer = None
label_encoder = None

# Constants from training
MAXLEN = 500
VOCAB_SIZE = 25000
MODEL_PATH = "models/final_model_tf.keras"
TOKENIZER_PATH = "models/tokenizer.json"
LABEL_ENCODER_PATH = "models/label_encoder.json"

@app.on_event("startup")
async def startup_event():
    """Load model and artifacts on startup."""
    global model, tokenizer, label_encoder
    
    # 1. Load Tokenizer
    if os.path.exists(TOKENIZER_PATH):
        try:
            with open(TOKENIZER_PATH, 'r') as f:
                tokenizer_data = json.load(f)
                tokenizer = tokenizer_from_json(json.dumps(tokenizer_data))
            logger.info("Tokenizer loaded.")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
    else:
        logger.warning(f"Tokenizer not found at {TOKENIZER_PATH}")

    # 2. Load Label Encoder
    if os.path.exists(LABEL_ENCODER_PATH):
        try:
            with open(LABEL_ENCODER_PATH, 'r') as f:
                classes = json.load(f)
                label_encoder = LabelEncoder()
                label_encoder.classes_ = np.array(classes)
            logger.info("Label Encoder loaded.")
        except Exception as e:
            logger.error(f"Failed to load label encoder: {e}")
    else:
        logger.warning(f"Label encoder not found at {LABEL_ENCODER_PATH}")

    # 3. Load Model
    # We need to define AttentionLayer class if it was saved with one and not in custom_objects automatically
    # In train_model.py we used a custom layer.
    class AttentionLayer(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
        def build(self, input_shape):
            self.W = self.add_weight(shape=(input_shape[-1],), initializer='glorot_uniform', trainable=True)
            super().build(input_shape)
        def call(self, inputs):
            scores = tf.tanh(inputs)
            scores = tf.tensordot(scores, self.W, axes=[2,0])
            weights = tf.nn.softmax(scores, axis=1)
            context = tf.reduce_sum(inputs * tf.expand_dims(weights, -1), axis=1)
            return context
        def get_config(self):
            return super().get_config()

    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH, custom_objects={"AttentionLayer": AttentionLayer})
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    else:
        logger.warning(f"Model not found at {MODEL_PATH}")

def predict_role_helper(text: str):
    """Helper to predict role from text."""
    if not model or not tokenizer or not label_encoder:
        return None
    
    # Preprocess
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAXLEN, padding='post', truncating='post')
    
    # Predict
    probs = model.predict(padded)
    top_idx = np.argsort(probs, axis=1)[:, ::-1][:, :3] # Top 3
    top_probs = np.take_along_axis(probs, top_idx, axis=1)[0]
    top_labels = label_encoder.inverse_transform(top_idx.flatten())
    
    return {
        "top_role": top_labels[0],
        "top_3_roles": top_labels.tolist(),
        "probabilities": [float(p) for p in top_probs]
    }

@app.post("/extract_text")
async def extract_text_endpoint(file: UploadFile = File(...)):
    """Extract text from uploaded PDF or Image."""
    contents = await file.read()
    filename = file.filename or "unknown"
    ext = os.path.splitext(filename)[1].lower()
    
    text = extract_text_from_file(file_bytes=contents, ext=ext)
    return {"filename": filename, "extracted_text": text}

@app.post("/score_resume")
async def score_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):
    """
    1. Extract text from Resume.
    2. Calculate ATS Score against Job Description.
    3. Predict Job Role.
    """
    contents = await resume.read()
    filename = resume.filename or "unknown"
    ext = os.path.splitext(filename)[1].lower()
    
    # 1. Extract
    resume_text = extract_text_from_file(file_bytes=contents, ext=ext)
    if not resume_text:
        raise HTTPException(status_code=400, detail="Could not extract text from resume.")
        
    # 2. ATS Score
    ats_result = calculate_ats_score(resume_text, job_description)
    
    # 3. Predict Role
    role_prediction = predict_role_helper(resume_text)
    
    return {
        "filename": filename,
        "ats_score": ats_result,
        "role_prediction": role_prediction,
        "extracted_text_snippet": resume_text[:200] + "..."
    }

@app.get("/")
def read_root():
    return {"message": "Resume Analyzer API is running. Use /docs for Swagger UI."}

@app.post("/parse_resume")
async def parse_resume_endpoint(file: UploadFile = File(...)):
    """
    Extract structured fields from resume using Spacy and Regex.
    Returns: Name, Email, Phone, Skills, Section snippets.
    """
    contents = await file.read()
    filename = file.filename or "unknown"
    ext = os.path.splitext(filename)[1].lower()
    
    text = extract_text_from_file(file_bytes=contents, ext=ext)
    if not text:
        raise HTTPException(status_code=400, detail="Could not extract text from resume.")
        
    parsed_data = parse_resume(text)
    
    return {
        "filename": filename,
        "parsed_data": parsed_data,
        "extracted_text_debug": text
    }

