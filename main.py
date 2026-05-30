from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import os
import uvicorn

# Fix keras loading issue
import keras
from keras.layers import InputLayer
from keras import backend as K

app = FastAPI(title="ECG AI Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
API_KEY    = os.environ.get("API_KEY", "ecg-secret-key-123")
MODEL_PATH = os.environ.get("MODEL_PATH", "ECG_CNN_Model.h5")

# ─────────────────────────────────────────────
#  LOAD MODEL — with compatibility fix
# ─────────────────────────────────────────────
print("Loading ECG CNN model...")
try:
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        custom_objects=None
    )
    INPUT_SHAPE    = model.input_shape   # (None, 224, 224, 3)
    OUTPUT_CLASSES = model.output_shape[-1]
    print(f"✅ Model loaded! Input: {INPUT_SHAPE} Output: {model.output_shape}")
except Exception as e:
    print(f"❌ Model load error: {e}")
    # Try legacy loader
    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
            options=tf.saved_model.LoadOptions(
                experimental_io_device='/job:localhost'
            )
        )
        INPUT_SHAPE    = model.input_shape
        OUTPUT_CLASSES = model.output_shape[-1]
        print(f"✅ Model loaded with legacy loader! Input: {INPUT_SHAPE}")
    except Exception as e2:
        print(f"❌ Both loaders failed: {e2}")
        model = None
        INPUT_SHAPE    = (None, 224, 224, 3)
        OUTPUT_CLASSES = 5

# ─────────────────────────────────────────────
#  ECG LABELS
# ─────────────────────────────────────────────
ECG_LABELS_5 = [
    "Normal",
    "Supraventricular Ectopy",
    "Ventricular Ectopy",
    "Fusion Beat",
    "Unclassifiable",
]

ECG_LABELS_4 = [
    "Normal",
    "Atrial Fibrillation",
    "Other Rhythm",
    "Noisy Recording",
]

ECG_LABELS_2 = [
    "Normal",
    "Abnormal",
]

# Pick label set based on output classes
if OUTPUT_CLASSES == 2:
    ECG_LABELS = ECG_LABELS_2
elif OUTPUT_CLASSES == 4:
    ECG_LABELS = ECG_LABELS_4
else:
    ECG_LABELS = ECG_LABELS_5

RISK_MAP = {
    "Normal":                    "low",
    "Supraventricular Ectopy":   "medium",
    "Ventricular Ectopy":        "high",
    "Fusion Beat":               "medium",
    "Unclassifiable":            "medium",
    "Atrial Fibrillation":       "high",
    "Other Rhythm":              "medium",
    "Noisy Recording":           "low",
    "Abnormal":                  "high",
}

DETAILS_MAP = {
    "Normal":                    "Your ECG looks normal. No abnormalities detected. Keep up your healthy lifestyle.",
    "Supraventricular Ectopy":   "An irregular beat from above the ventricles. Usually harmless but worth monitoring.",
    "Ventricular Ectopy":        "An irregular beat from the lower chambers. Consult a doctor if frequent.",
    "Fusion Beat":               "A combined beat pattern detected. Please consult a cardiologist.",
    "Unclassifiable":            "Pattern could not be classified clearly. Please repeat ECG or see a doctor.",
    "Atrial Fibrillation":       "Irregular rapid rhythm detected. Increases stroke risk. See a cardiologist immediately.",
    "Other Rhythm":              "An unusual heart rhythm detected. Please consult your doctor.",
    "Noisy Recording":           "The ECG signal had too much noise. Please retake the reading.",
    "Abnormal":                  "An abnormal ECG pattern was detected. Please consult a healthcare professional.",
}

# ─────────────────────────────────────────────
#  REQUEST MODELS
# ─────────────────────────────────────────────
class ECGRequest(BaseModel):
    ecg_signal: list
    metadata: dict = {}

class ChatRequest(BaseModel):
    message: str
    session_history: list = []
    ecg_context: dict = None
    user_id: str = ""

# ─────────────────────────────────────────────
#  CONVERT ECG SIGNAL → 224x224x3 IMAGE
#  Plots the signal as a grayscale waveform image
# ─────────────────────────────────────────────
def signal_to_image(ecg_signal: list) -> np.ndarray:
    signal = np.array(ecg_signal, dtype=np.float32)

    # Normalize signal to 0-1
    s_min, s_max = signal.min(), signal.max()
    if s_max - s_min > 0:
        signal = (signal - s_min) / (s_max - s_min)

    # Create 224x224 blank white image
    img_size = 224
    image = np.ones((img_size, img_size), dtype=np.float32)

    # Plot signal as waveform on image
    n = len(signal)
    for i in range(n - 1):
        x1 = int((i / n) * img_size)
        x2 = int(((i + 1) / n) * img_size)
        y1 = int((1 - signal[i]) * (img_size - 1))
        y2 = int((1 - signal[i + 1]) * (img_size - 1))

        # Clamp to image bounds
        y1 = max(0, min(img_size - 1, y1))
        y2 = max(0, min(img_size - 1, y2))

        # Draw line between points
        steps = max(abs(x2 - x1), abs(y2 - y1), 1)
        for step in range(steps + 1):
            t = step / steps
            px = int(x1 + t * (x2 - x1))
            py = int(y1 + t * (y2 - y1))
            if 0 <= px < img_size and 0 <= py < img_size:
                image[py, px] = 0.0  # black line

    # Convert to RGB (3 channels) and add batch dim → (1, 224, 224, 3)
    image_rgb = np.stack([image, image, image], axis=-1)
    image_rgb = np.expand_dims(image_rgb, axis=0)

    return image_rgb

# ─────────────────────────────────────────────
#  ENDPOINT 1 — /predict
# ─────────────────────────────────────────────
@app.post("/predict")
async def predict(req: ECGRequest, x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not req.ecg_signal or len(req.ecg_signal) < 10:
        raise HTTPException(status_code=400, detail="ecg_signal needs at least 10 values")

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Convert signal to image
        image = signal_to_image(req.ecg_signal)

        # Run prediction
        predictions = model.predict(image, verbose=0)
        pred_index  = int(np.argmax(predictions[0]))
        confidence  = float(predictions[0][pred_index])

        condition = ECG_LABELS[pred_index] if pred_index < len(ECG_LABELS) else f"Class_{pred_index}"
        risk      = RISK_MAP.get(condition, "medium")
        details   = DETAILS_MAP.get(condition, "Please consult a healthcare professional.")

        return {
            "condition":  condition,
            "confidence": round(confidence, 4),
            "risk_level": risk,
            "details":    details,
            "all_scores": {
                ECG_LABELS[i]: round(float(predictions[0][i]), 4)
                for i in range(min(len(ECG_LABELS), len(predictions[0])))
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ─────────────────────────────────────────────
#  ENDPOINT 2 — /chat
# ─────────────────────────────────────────────
@app.post("/chat")
async def chat(req: ChatRequest, x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    msg  = req.message.lower().strip()
    ecg  = req.ecg_context or {}
    condition  = ecg.get("condition", "")
    risk       = ecg.get("risk_level", "")
    confidence = ecg.get("confidence", 0)

    if condition:
        if any(w in msg for w in ["mean", "result", "what", "diagnosis", "condition"]):
            reply = (
                f"Your ECG shows **{condition}** with "
                f"{round(confidence * 100)}% confidence. "
                f"This is a **{risk} risk** finding. "
                f"{DETAILS_MAP.get(condition, '')}"
            )
        elif any(w in msg for w in ["serious", "dangerous", "bad", "worry", "scared"]):
            if risk == "high":
                reply = f"{condition} is serious. Please contact a cardiologist or go to emergency room immediately."
            elif risk == "medium":
                reply = f"{condition} is moderate. Schedule an appointment with your doctor soon."
            else:
                reply = f"{condition} is generally not serious. Continue monitoring your heart health regularly."
        elif any(w in msg for w in ["doctor", "hospital", "when", "should i"]):
            if risk == "high":
                reply = "Based on your ECG, please see a cardiologist today or go to emergency room immediately."
            elif risk == "medium":
                reply = "Schedule a doctor appointment within the next few days to discuss your ECG result."
            else:
                reply = "A routine checkup with your doctor is recommended. No immediate urgency."
        elif any(w in msg for w in ["lifestyle", "diet", "food", "exercise", "improve"]):
            reply = (
                "For better heart health: eat a low-sodium diet, exercise 30 min/day, "
                "avoid smoking, limit alcohol, manage stress, sleep 7-8 hours, "
                "and monitor your blood pressure regularly."
            )
        elif any(w in msg for w in ["medicine", "medication", "tablet", "drug", "treatment"]):
            reply = f"I cannot prescribe medications. For {condition}, your doctor will recommend the right treatment based on your full medical history."
        else:
            reply = (
                f"Your ECG shows {condition} ({risk} risk). "
                "Ask me what it means, how serious it is, when to see a doctor, or what lifestyle changes to make."
            )
    else:
        if any(w in msg for w in ["hi", "hello", "hey"]):
            reply = "Hello! I am your ECG health assistant. Take an ECG reading first so I can give you personalized advice."
        elif any(w in msg for w in ["heart attack", "chest pain"]):
            reply = "If you are experiencing chest pain, call emergency services (112 or 911) immediately. Do not wait."
        else:
            reply = "I am your ECG health assistant. Please take an ECG reading first so I can give you personalized guidance."

    return {"reply": reply}

# ─────────────────────────────────────────────
#  ENDPOINT 3 — /health
# ─────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status":         "ok",
        "model_loaded":   model is not None,
        "input_shape":    str(INPUT_SHAPE),
        "output_classes": OUTPUT_CLASSES,
        "labels":         ECG_LABELS,
    }

@app.get("/")
async def root():
    return {"message": "ECG AI Server is running"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)
