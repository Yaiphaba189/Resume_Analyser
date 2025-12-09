"""
Resume Job-Role Classifier (Improved Version)
Fixes included:
 - No data leakage (tokenizer trained ONLY on training texts)
 - Token index clipping to avoid out-of-range embedding errors
 - Modern Keras saving format (.keras)
 - Safe custom layer loading
 - More stable training setup
"""

import os
import json
import random
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, GlobalMaxPooling1D, Concatenate,
    Dense, Dropout, LSTM, Bidirectional
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ---------------------------------------
# Reproducibility
# ---------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------------------
# Config
# ---------------------------------------
# Note: Run this script from root (backend/) or adjust paths
DATASET_CSV = "data/dataset.csv"
RESUMES_CSV = "data/resumes_converted.csv"

TEXT_COL = "Resume"
LABEL_COL = "Role"
RESUME_ID_COL = "id"

VOCAB_SIZE = 25000
MAXLEN = 500
EMBEDDING_DIM = 200
BATCH_SIZE = 64
EPOCHS = 12

MODEL_TYPE = "textcnn"   # "textcnn" or "bilstm"
USE_ATTENTION = True
TOP_K = 3

MODEL_SAVE_PATH = "models/best_model.keras"
TOKENIZER_SAVE = "models/tokenizer.json"
LABEL_ENCODER_SAVE = "models/label_encoder.json"
PREDICTIONS_OUT = "data/resume_predictions.csv"
FINAL_MODEL_DIR = "models/final_model_tf.keras"

# ---------------------------------------
# Utility: Read & Clean Data
# ---------------------------------------
def read_and_clean():
    df = pd.read_csv(DATASET_CSV)
    resumes = pd.read_csv(RESUMES_CSV)

    df = df.dropna(subset=[TEXT_COL, LABEL_COL]).reset_index(drop=True)
    resumes = resumes.dropna(subset=[TEXT_COL]).reset_index(drop=True)

    def normalize(s):
        s = str(s).replace("\n", " ").replace("\r", " ")
        return " ".join(s.split())

    df[TEXT_COL] = df[TEXT_COL].map(normalize)
    resumes[TEXT_COL] = resumes[TEXT_COL].map(normalize)

    return df, resumes

# ---------------------------------------
# Tokenizer + Safe Sequence Processing
# ---------------------------------------
def build_tokenizer(texts, vocab_size=VOCAB_SIZE, oov_token="<OOV>"):
    tok = Tokenizer(
        num_words=vocab_size,
        oov_token=oov_token,
        lower=True,
        filters='''!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n'''
    )
    tok.fit_on_texts(texts)
    return tok

def texts_to_padded(tokenizer, texts, maxlen=MAXLEN, vocab_size=VOCAB_SIZE, oov_token="<OOV>"):
    seqs = tokenizer.texts_to_sequences(texts)
    oov_index = tokenizer.word_index.get(oov_token, 1)

    clipped = []
    for s in seqs:
        clipped.append([token if token < vocab_size else oov_index for token in s])

    return pad_sequences(clipped, maxlen=maxlen, padding='post', truncating='post')

# ---------------------------------------
# Models
# ---------------------------------------
def build_text_cnn(vocab_size, num_classes, embedding_dim=EMBEDDING_DIM):
    inp = Input(shape=(MAXLEN,), dtype='int32')
    x = Embedding(vocab_size, embedding_dim)(inp)

    convs = []
    for fsz in (3,4,5):
        c = Conv1D(128, fsz, activation='relu')(x)
        p = GlobalMaxPooling1D()(c)
        convs.append(p)

    x = Concatenate()(convs)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    out = Dense(num_classes, activation='softmax')(x)

    return Model(inp, out)

# --- Attention Layer ---
class AttentionLayer(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1],),
            initializer='glorot_uniform',
            trainable=True
        )

    def call(self, inputs):
        scores = tf.tanh(inputs)
        scores = tf.tensordot(scores, self.W, axes=[2,0])
        weights = tf.nn.softmax(scores, axis=1)
        context = tf.reduce_sum(inputs * tf.expand_dims(weights, -1), axis=1)
        return context

def build_bilstm(vocab_size, num_classes):
    inp = Input(shape=(MAXLEN,))
    x = Embedding(vocab_size, EMBEDDING_DIM)(inp)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = AttentionLayer()(x) if USE_ATTENTION else GlobalMaxPooling1D()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    out = Dense(num_classes, activation='softmax')(x)
    return Model(inp, out)

# ---------------------------------------
# Training
# ---------------------------------------
def train_and_evaluate(X_train, y_train, X_val, y_val, vocab_size, num_classes):
    model = build_text_cnn(vocab_size, num_classes) if MODEL_TYPE=="textcnn" \
            else build_bilstm(vocab_size, num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    cw = {int(c): float(w) for c, w in zip(np.unique(y_train), class_weights)}

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_SAVE_PATH, monitor="val_loss", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1, min_lr=1e-6)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=cw
    )

    return model, history

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.savefig('training_history.png')
    print("Training history plot saved as training_history.png")

# ---------------------------------------
# Prediction (Top-K)
# ---------------------------------------
def predict_topk(model, tokenizer, texts, label_encoder, top_k=3):
    X = texts_to_padded(tokenizer, texts)
    probs = model.predict(X)
    top_idx = np.argsort(probs, axis=1)[:, ::-1][:, :top_k]
    top_probs = np.take_along_axis(probs, top_idx, axis=1)
    labels = label_encoder.inverse_transform(top_idx.flatten()).reshape(top_idx.shape)

    results = []
    for i in range(len(texts)):
        results.append({
            "top_k_labels": list(labels[i]),
            "top_k_probs": list(top_probs[i])
        })
    return results, probs

# ---------------------------------------
# Save/Load
# ---------------------------------------
def save_tokenizer(tok):
    with open(TOKENIZER_SAVE, "w", encoding="utf-8") as f:
        f.write(tok.to_json())

def save_label_encoder(le):
    with open(LABEL_ENCODER_SAVE, "w") as f:
        json.dump(le.classes_.tolist(), f)

# ---------------------------------------
# Main
# ---------------------------------------
def main():
    df, resumes = read_and_clean()
    texts = df[TEXT_COL].tolist()
    labels = df[LABEL_COL].tolist()

    print("Total training samples:", len(df))

    le = LabelEncoder()
    y = le.fit_transform(labels)
    num_classes = len(le.classes_)
    save_label_encoder(le)

    print("Building tokenizer on TRAINING texts (no leakage)...")
    tokenizer = build_tokenizer(texts)
    save_tokenizer(tokenizer)

    print("Converting to padded sequences...")
    X = texts_to_padded(tokenizer, texts)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.12, random_state=42, stratify=y
    )

    print("Training...")    # Train
    model, history = train_and_evaluate(X_train, y_train, X_val, y_val, vocab_size=VOCAB_SIZE, num_classes=num_classes)

    # Plot history
    plot_history(history)

    # Reload best model safely
    custom = {"AttentionLayer": AttentionLayer}
    model = load_model(MODEL_SAVE_PATH, custom_objects=custom)

    # Evaluate
    print("\nValidation Report:")
    preds = np.argmax(model.predict(X_val), axis=1)
    print(classification_report(y_val, preds, target_names=le.classes_))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, preds))

    # Predict resumes
    print("\nPredicting resumes...")
    resume_texts = resumes[TEXT_COL].tolist()
    results, all_probs = predict_topk(model, tokenizer, resume_texts, le, top_k=TOP_K)

    out = []
    for i, r in enumerate(results):
        out.append({
            "input_index": i,
            "id": resumes.loc[i, RESUME_ID_COL] if RESUME_ID_COL in resumes.columns else "",
            "predicted_role": r["top_k_labels"][0],
            "top_k_roles": "|".join(r["top_k_labels"]),
            "top_k_probs": "|".join([f"{p:.4f}" for p in r["top_k_probs"]])
        })

    pd.DataFrame(out).to_csv(PREDICTIONS_OUT, index=False)
    print("Saved predictions to", PREDICTIONS_OUT)

    # Save TF model
    model.save(FINAL_MODEL_DIR)
    print("Saved final TF model to", FINAL_MODEL_DIR)


if __name__ == "__main__":
    main()
