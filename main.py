from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from PIL import Image
import io
import numpy as np
import joblib
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2

from sqlalchemy.orm import Session
from jose import jwt, JWTError

from decision_engine import decide_action
from recommendation import recommend_places

from db import SessionLocal, engine, Base
from models_db import User, Analysis
from auth import hash_password, verify_password, create_access_token


# -------------------------------
# App
# -------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://smart-circular-ai-frontend.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# -------------------------------
# DB init
# -------------------------------

Base.metadata.create_all(bind=engine)


# -------------------------------
# DB dependency
# -------------------------------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -------------------------------
# JWT dependency
# -------------------------------

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

SECRET_KEY = "change_this_secret_key"
ALGORITHM = "HS256"


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter(User.email == email).first()

    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user


# -------------------------------
# Human face gate (FAST filter)
# -------------------------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def contains_human_face(pil_image):
    img = np.array(pil_image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return len(faces) > 0


# -------------------------------
# CNN loader (condition models)
# -------------------------------

def load_cnn_model(path):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


cnn_models = {
    "laptop": load_cnn_model("models/laptop_model.pt"),
    "mobile": load_cnn_model("models/mobile_model.pt")
}

condition_classes = ["Good", "Average", "Bad"]


# -------------------------------
# Product type classifier
# -------------------------------

type_model = models.resnet18(weights=None)
type_model.fc = nn.Linear(type_model.fc.in_features, 2)
type_model.load_state_dict(torch.load("models/type_model.pt", map_location="cpu"))
type_model.eval()

type_classes = ["laptop", "mobile"]


# -------------------------------
# Phase-2 & Phase-3 models
# -------------------------------

lifecycle_model = joblib.load("lifecycle_model.pkl")
pricing_model   = joblib.load("pricing_model.pkl")


# -------------------------------
# Image preprocessing
# -------------------------------

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])


# -------------------------------
# Health
# -------------------------------

@app.get("/")
def home():
    return {"status": "Circular Economy AI backend running"}


# =========================================================
# AUTH
# =========================================================

@app.post("/auth/register")
def register(email: str, password: str, db: Session = Depends(get_db)):

    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=400, detail="Email already exists")

    user = User(
        email=email,
        hashed_password=hash_password(password)
    )

    db.add(user)
    db.commit()

    return {"message": "registered"}


@app.post("/auth/login")
def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):

    user = db.query(User).filter(User.email == form.username).first()

    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": user.email})

    return {"access_token": token, "token_type": "bearer"}


# =========================================================
# OLD decision API
# =========================================================

@app.post("/decision")
def get_decision(
    condition_score: float,
    remaining_life_months: float,
    price: float,
    lat: float,
    lon: float
):

    decision = decide_action(
        condition_score,
        remaining_life_months,
        price
    )

    places = recommend_places(
        decision["action"],
        lat,
        lon
    )

    return {
        "action": decision["action"],
        "reason": decision["reason"],
        "recommended_places": places
    }


# =========================================================
# MAIN AI PIPELINE
# =========================================================

@app.post("/analyze-product")
async def analyze_product(
    image: UploadFile = File(...),
    age_months: int = Form(...),
    usage_level: int = Form(...),
    base_price: float = Form(...),
    demand_level: int = Form(...),
    lat: float = Form(...),
    lon: float = Form(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):

    # -------------------------
    # Read image
    # -------------------------

    contents = await image.read()

    # size limit (optional but ok)
    if len(contents) > 1_000_000:
        raise HTTPException(status_code=400, detail="Image too large")

    # IMPORTANT: reset pointer (safety for future use)
    image.file.seek(0)

    img = Image.open(io.BytesIO(contents)).convert("RGB")

    # -------------------------
    # HUMAN IMAGE REJECTION
    # -------------------------

    if contains_human_face(img):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "human_image_detected",
                "message": "Please upload only a product image (laptop or mobile)."
            }
        )
    # ---------------------------------
# Signature / blank image rejection
# ---------------------------------

    if looks_like_signature_or_blank(img):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_image",
                "message": "Uploaded image looks like a document or signature. Please upload a real product photo."
            }
        )

    def looks_like_signature_or_blank(pil_image):
        img = np.array(pil_image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # normalize
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # binary image (separate ink from background)
        _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # percentage of foreground pixels
        foreground_ratio = np.count_nonzero(th) / th.size

        return foreground_ratio < 0.05


    img_tensor = img_transform(img).unsqueeze(0)

    # -------------------------
    # Phase 0 – product type
    # -------------------------

    with torch.no_grad():
        type_out = type_model(img_tensor)
        type_probs = torch.softmax(type_out, dim=1)[0]

    top2 = torch.topk(type_probs, 2)

    best_prob = float(top2.values[0])
    second_prob = float(top2.values[1])
    margin = best_prob - second_prob

    detected_category = type_classes[int(top2.indices[0])]
    category_confidence = best_prob

    # -------------------------
    # REJECTION GATE
    # -------------------------

    if category_confidence < 0.75 or margin < 0.25:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_image",
                "message": "Please upload a clear image of a laptop or a mobile device only.",
                "confidence": round(category_confidence, 3),
                "margin": round(margin, 3)
            }
        )

    # -------------------------
    # Phase 1 – condition
    # -------------------------

    condition_model = cnn_models[detected_category]

    with torch.no_grad():
        out = condition_model(img_tensor)
        probs = torch.softmax(out, dim=1)[0]

    condition_score = float(probs.max().item())
    condition_class = condition_classes[int(probs.argmax().item())]

    # -------------------------
    # Phase 2 – lifecycle
    # -------------------------

    X_life = np.array([[condition_score, age_months, usage_level]])
    remaining_life = float(lifecycle_model.predict(X_life)[0])
    remaining_life = max(0.0, remaining_life)

    # -------------------------
    # Phase 3 – pricing
    # -------------------------

    X_price = np.array([[
        condition_score,
        remaining_life,
        age_months,
        base_price,
        demand_level
    ]])

    recommended_price = float(pricing_model.predict(X_price)[0])
    recommended_price = max(500.0, recommended_price)

    # -------------------------
    # Phase 4 – decision
    # -------------------------

    decision = decide_action(
        condition_score,
        remaining_life,
        recommended_price
    )

    # -------------------------
    # Phase 5 – recommendation
    # -------------------------

    places = recommend_places(
        decision["action"],
        lat,
        lon
    )

    # -------------------------
    # Save history
    # -------------------------

    record = Analysis(
        user_id=user.id,
        detected_category=detected_category,
        condition=condition_class,
        condition_score=condition_score,
        remaining_life=remaining_life,
        price=recommended_price,
        action=decision["action"]
    )

    db.add(record)
    db.commit()

    # -------------------------
    # Response
    # -------------------------

    return {
        "detected_category": detected_category,
        "category_confidence": round(category_confidence, 3),

        "condition": condition_class,
        "condition_score": round(condition_score, 3),

        "remaining_life_months": round(remaining_life, 2),
        "recommended_price": round(recommended_price, 2),

        "action": decision["action"],
        "reason": decision["reason"],
        "recommended_places": places
    }


# =========================================================
# USER HISTORY
# =========================================================

@app.get("/me/history")
def my_history(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):

    rows = db.query(Analysis)\
        .filter(Analysis.user_id == user.id)\
        .order_by(Analysis.created_at.desc())\
        .all()

    return rows
