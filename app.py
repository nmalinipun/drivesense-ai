import os
import csv
import shutil
from pathlib import Path
from datetime import datetime

import joblib
import librosa
import numpy as np
import streamlit as st

try:
    from google import genai
except Exception:
    genai = None


# ==============================================================================
# 1. PAGE CONFIG
# ==============================================================================
st.set_page_config(
    page_title="DriveSense AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================================================================
# 2. GLOBAL CONFIG
# ==============================================================================
TARGET_SR = 22050
WINDOW_SECONDS = 3.0
WINDOW_SAMPLES = int(TARGET_SR * WINDOW_SECONDS)
CONFIDENCE_THRESHOLD = 0.60

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "project_milo_final_classifier.joblib"
ENCODER_PATH = BASE_DIR / "project_milo_label_encoder.joblib"
REFERENCE_ROOT = BASE_DIR / "reference_audio"
FEEDBACK_ROOT = BASE_DIR / "feedback_audio"
FEEDBACK_LOG_CSV = BASE_DIR / "feedback_log_project_milo.csv"
TEMP_UPLOAD_DIR = BASE_DIR / "temp_uploads"

EXPECTED_CLASSES = [
    "bad_brakes",
    "bad_ignition",
    "low_engine_oil",
    "low_oil_power_steering",
    "low_oil_power_steering_serpentine_belt",
    "low_oil_serpentine_belt",
    "normal_brakes",
    "normal_engine_idle",
    "normal_engine_start",
    "power_steering_idle",
    "power_steering_serpentine_belt",
    "serpentine_belt_idle",
]

DISPLAY_NAMES = {
    "bad_brakes": "Bad Brake Sound",
    "bad_ignition": "Bad Ignition Sound",
    "low_engine_oil": "Low Engine Oil",
    "low_oil_power_steering": "Low Oil / Power Steering",
    "low_oil_power_steering_serpentine_belt": "Low Oil / Power Steering / Serpentine Belt",
    "low_oil_serpentine_belt": "Low Oil / Serpentine Belt",
    "normal_brakes": "Normal Brakes",
    "normal_engine_idle": "Normal Engine Idle",
    "normal_engine_start": "Normal Engine Start",
    "power_steering_idle": "Power Steering at Idle",
    "power_steering_serpentine_belt": "Power Steering / Serpentine Belt",
    "serpentine_belt_idle": "Serpentine Belt at Idle",
}

COMPONENT_MAP = {
    "bad_brakes": "Braking System",
    "normal_brakes": "Braking System",
    "bad_ignition": "Ignition / Powertrain",
    "normal_engine_idle": "Engine / Powertrain",
    "normal_engine_start": "Engine / Starting System",
    "low_engine_oil": "Lubrication System",
    "low_oil_power_steering": "Power Steering / Lubrication",
    "low_oil_power_steering_serpentine_belt": "Power Steering / Belt / Lubrication",
    "low_oil_serpentine_belt": "Serpentine Belt / Lubrication",
    "power_steering_idle": "Power Steering",
    "power_steering_serpentine_belt": "Power Steering / Belt",
    "serpentine_belt_idle": "Serpentine Belt",
}


# ==============================================================================
# 3. HELPERS
# ==============================================================================
def pretty_label(label: str) -> str:
    return DISPLAY_NAMES.get(label, label.replace("_", " ").title())


def sound_icon_svg(size: int = 74) -> str:
    inner = int(size * 0.55)
    return f"""
    <div style="
        width:{size}px;
        height:{size}px;
        border-radius:22px;
        display:flex;
        align-items:center;
        justify-content:center;
        background:linear-gradient(135deg,#2563eb 0%, #3b82f6 100%);
        box-shadow:0 14px 34px rgba(37,99,235,0.30);
        border:1px solid rgba(255,255,255,0.12);
    ">
        <svg width="{inner}" height="{inner}" viewBox="0 0 24 24" fill="none"
             xmlns="http://www.w3.org/2000/svg">
            <rect x="2" y="9" width="3" height="6" rx="1.5" fill="white" opacity="0.95"/>
            <rect x="7" y="6" width="3" height="12" rx="1.5" fill="white" opacity="0.95"/>
            <rect x="12" y="3" width="3" height="18" rx="1.5" fill="white" opacity="0.95"/>
            <rect x="17" y="7" width="3" height="10" rx="1.5" fill="white" opacity="0.95"/>
        </svg>
    </div>
    """


def ensure_dirs() -> None:
    TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    FEEDBACK_ROOT.mkdir(parents=True, exist_ok=True)
    REFERENCE_ROOT.mkdir(parents=True, exist_ok=True)


def load_model_and_encoder():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH.name}")
        st.stop()
    if not ENCODER_PATH.exists():
        st.error(f"Encoder file not found: {ENCODER_PATH.name}")
        st.stop()

    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, encoder


def audio_to_temp_file(uploaded_file, filename_hint: str | None = None) -> Path:
    ensure_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suffix = ".wav"
    original_name = getattr(uploaded_file, "name", "") or ""
    if "." in original_name:
        suffix = Path(original_name).suffix.lower() or ".wav"
    if filename_hint:
        safe_name = filename_hint
    else:
        safe_name = f"audio_{timestamp}{suffix}"
    temp_path = TEMP_UPLOAD_DIR / safe_name

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return temp_path


def extract_features(file_path: Path) -> np.ndarray:
    y, sr = librosa.load(str(file_path), sr=TARGET_SR, mono=True)

    if len(y) < WINDOW_SAMPLES:
        y = np.pad(y, (0, WINDOW_SAMPLES - len(y)))
    else:
        y = y[:WINDOW_SAMPLES]

    y = librosa.util.normalize(y)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    feature_vector = np.hstack([
        np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
        np.mean(chroma, axis=1), np.std(chroma, axis=1),
        np.mean(mel, axis=1), np.std(mel, axis=1),
        np.mean(zcr, axis=1), np.std(zcr, axis=1),
        np.mean(rms, axis=1), np.std(rms, axis=1),
        np.mean(centroid, axis=1), np.std(centroid, axis=1),
        np.mean(bandwidth, axis=1), np.std(bandwidth, axis=1),
        np.mean(rolloff, axis=1), np.std(rolloff, axis=1),
        np.mean(contrast, axis=1), np.std(contrast, axis=1),
    ]).astype(np.float32)

    return feature_vector.reshape(1, -1)


def predict_audio(file_path: Path, model, encoder):
    X = extract_features(file_path)

    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
    else:
        pred_idx = model.predict(X)[0]
        probs = np.zeros(len(encoder.classes_), dtype=float)
        probs[int(pred_idx)] = 1.0

    top_idx = np.argsort(probs)[::-1][:3]
    top_preds = []
    for i in top_idx:
        label = encoder.classes_[i]
        top_preds.append({
            "label": label,
            "display": pretty_label(label),
            "component": COMPONENT_MAP.get(label, "Vehicle System"),
            "prob": float(probs[i]),
        })

    primary = top_preds[0]
    return {
        "top_preds": top_preds,
        "primary_label": primary["label"],
        "primary_display": primary["display"],
        "primary_component": primary["component"],
        "confidence": primary["prob"],
    }


def get_reference_clips_for_class(class_name: str) -> list[Path]:
    class_dir = REFERENCE_ROOT / class_name
    if not class_dir.exists():
        return []
    clips = []
    for ext in ["*.wav", "*.mp3", "*.m4a", "*.aac", "*.flac"]:
        clips.extend(sorted(class_dir.glob(ext)))
    return clips[:5]


def save_feedback(
    uploaded_temp_path: Path | None,
    selected_class: str,
    selected_clips: list[str],
    vehicle_make: str,
    vehicle_model: str,
    vehicle_year: int,
    mileage: int,
) -> Path | None:
    if uploaded_temp_path is None or not uploaded_temp_path.exists():
        return None

    ensure_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    class_dir = FEEDBACK_ROOT / selected_class
    class_dir.mkdir(parents=True, exist_ok=True)

    new_name = f"{timestamp}_{uploaded_temp_path.name}"
    saved_path = class_dir / new_name
    shutil.copy2(uploaded_temp_path, saved_path)

    file_exists = FEEDBACK_LOG_CSV.exists()
    with open(FEEDBACK_LOG_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp",
                "saved_audio_path",
                "selected_class",
                "selected_reference_clips",
                "vehicle_make",
                "vehicle_model",
                "vehicle_year",
                "mileage",
            ])
        writer.writerow([
            datetime.now().isoformat(),
            str(saved_path),
            selected_class,
            "; ".join(selected_clips),
            vehicle_make,
            vehicle_model,
            vehicle_year,
            mileage,
        ])

    return saved_path


def generate_llm_summary(
    selected_class: str,
    top_preds: list[dict],
    vehicle_make: str,
    vehicle_model: str,
    vehicle_year: int,
    mileage: int,
) -> str:
    fallback = f"""
**Likely sound class:** {pretty_label(selected_class)}

**What it may indicate**
- Most likely related system: {COMPONENT_MAP.get(selected_class, "Vehicle system")}
- Vehicle context considered: {vehicle_year} {vehicle_make} {vehicle_model}, {mileage:,} miles
- This output is based on the uploaded sound and the selected closest reference class

**Recommended next checks**
- Confirm whether the sound is strongest at idle, start-up, braking, or steering
- Avoid aggressive driving until the issue is verified
- Inspect the related system first and compare with a mechanic evaluation if the sound worsens

**Practical note**
- This is a decision-support result, not a final mechanical diagnosis
""".strip()

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or genai is None:
        return fallback

    try:
        client = genai.Client(api_key=api_key)
        prompt = f"""
You are an automotive diagnostic assistant.

Create a short, structured, plain-English report using these inputs:
- Selected sound class: {pretty_label(selected_class)}
- Related vehicle system: {COMPONENT_MAP.get(selected_class, "Vehicle system")}
- Vehicle: {vehicle_year} {vehicle_make} {vehicle_model}
- Mileage: {mileage}
- Top 3 model hypotheses:
{chr(10).join([f"  - {p['display']} ({p['prob']:.1%})" for p in top_preds])}

Rules:
- Write only 3 sections:
1. Likely issue
2. What the user should check next
3. Caution level
- Keep it concise
- Do not claim certainty
- Do not mention probabilities explicitly
- Avoid overly technical wording
"""
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        text = getattr(response, "text", None)
        return text.strip() if text else fallback
    except Exception:
        return fallback


# ==============================================================================
# 4. SESSION STATE
# ==============================================================================
if "stage" not in st.session_state:
    st.session_state.stage = "input"

if "result" not in st.session_state:
    st.session_state.result = None

if "uploaded_temp_path" not in st.session_state:
    st.session_state.uploaded_temp_path = None

if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

if "selected_reference_class" not in st.session_state:
    st.session_state.selected_reference_class = None

if "selected_reference_clips" not in st.session_state:
    st.session_state.selected_reference_clips = []

if "saved_feedback_path" not in st.session_state:
    st.session_state.saved_feedback_path = None

if "llm_report" not in st.session_state:
    st.session_state.llm_report = None


# ==============================================================================
# 5. CAR DATA
# ==============================================================================
car_data = {
    "Audi": ["A3", "A4", "A6", "Q5", "Q7"],
    "BMW": ["3 Series", "5 Series", "X3", "X5", "7 Series"],
    "Chevrolet": ["Cruze", "Equinox", "Malibu", "Silverado", "Tahoe"],
    "Ford": ["Escape", "Explorer", "F-150", "Focus", "Fusion"],
    "Honda": ["Accord", "Civic", "CR-V", "Fit", "Pilot"],
    "Hyundai": ["Accent", "Elantra", "Santa Fe", "Sonata", "Tucson"],
    "Jeep": ["Cherokee", "Compass", "Grand Cherokee", "Renegade", "Wrangler"],
    "Kia": ["Forte", "Optima", "Sorento", "Soul", "Sportage"],
    "Lexus": ["ES350", "GX460", "IS250", "LS460", "RX350"],
    "Mazda": ["CX-30", "CX-5", "CX-9", "Mazda3", "Mazda6"],
    "Mercedes-Benz": ["C-Class", "E-Class", "GLC", "GLE", "S-Class"],
    "Nissan": ["Altima", "Murano", "Pathfinder", "Rogue", "Sentra"],
    "Subaru": ["Crosstrek", "Forester", "Impreza", "Legacy", "Outback"],
    "Toyota": ["Camry", "Corolla", "Highlander", "Prius", "RAV4"],
    "Volkswagen": ["Atlas", "Golf", "Jetta", "Passat", "Tiguan"],
}
year_options = list(range(2026, 1995, -1))


# ==============================================================================
# 6. CSS
# ==============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500;600&family=Syne:wght@600;700;800&display=swap');

html, body, .stApp{
    font-family:'Syne',sans-serif !important;
    background:
        radial-gradient(circle at top left, rgba(41,98,255,0.16), transparent 28%),
        radial-gradient(circle at top right, rgba(0,180,255,0.08), transparent 22%),
        linear-gradient(135deg, #081224 0%, #0c1730 45%, #132a57 100%) !important;
    color:#f1f5f9 !important;
}

header[data-testid="stHeader"],
div[data-testid="stToolbar"],
div[data-testid="stDecoration"],
#MainMenu,
footer{
    display:none !important;
}

.main .block-container{
    padding-top:1.2rem !important;
    padding-bottom:2.6rem !important;
    max-width:1450px !important;
    padding-left:2rem !important;
    padding-right:2rem !important;
}

h1,h2,h3,h4,h5,h6,p,li,b,strong{
    color:#f1f5f9 !important;
}

section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0b1220 0%,#0f172a 100%) !important;
    border-right:1px solid rgba(255,255,255,0.07) !important;
}

/* Sidebar logo */
.ds-logo-wrap{
    display:flex;
    align-items:center;
    gap:12px;
    padding:8px 6px 10px 6px;
}
.ds-logo-name{
    font-size:19px;
    font-weight:800;
    color:#f8fbff;
    line-height:1;
}
.ds-logo-sub{
    margin-top:6px;
    font-size:11px;
    color:#86a3d8;
    font-family:'IBM Plex Mono', monospace;
    text-transform:uppercase;
    letter-spacing:1.4px;
}

/* Header */
.ds-header-wrap{
    display:flex;
    align-items:center;
    gap:18px;
    margin-bottom:14px;
}
.ds-page-title{
    font-size: clamp(2.5rem, 5vw, 4.3rem);
    line-height:0.95;
    font-weight:800;
    color:#f8fbff;
    letter-spacing:-1.4px;
}
.ds-page-subtitle{
    margin-top:10px;
    color:#7f97c3;
    font-size:13px;
    font-family:'IBM Plex Mono', monospace;
    letter-spacing:1.8px;
    text-transform:uppercase;
}

/* Section titles */
.ds-step-badge{
    display:inline-block;
    margin-top:10px;
    margin-bottom:12px;
    padding:8px 16px;
    border-radius:999px;
    border:1px solid rgba(74,132,255,0.32);
    background:rgba(29,78,216,0.10);
    color:#9fc3ff;
    font-family:'IBM Plex Mono', monospace;
    font-size:13px;
    font-weight:700;
    letter-spacing:1px;
}
.ds-section{
    font-size: clamp(2rem, 4vw, 3.2rem);
    font-weight:800;
    line-height:1;
    color:#f8fbff;
    letter-spacing:-1px;
    margin-bottom:8px;
}
.ds-section-sub{
    font-size:14px;
    color:#8ea6d6;
    margin-bottom:18px;
    font-weight:600;
}

/* Inputs */
.stSelectbox label,
.stSlider label,
.stRadio label{
    font-size:12px !important;
    font-weight:700 !important;
    color:#9fb3d9 !important;
    text-transform:uppercase !important;
    letter-spacing:1.4px !important;
    font-family:'IBM Plex Mono',monospace !important;
}

div[data-baseweb="select"] > div{
    background:#182744 !important;
    border:1px solid rgba(255,255,255,0.10) !important;
    border-radius:16px !important;
    min-height:54px !important;
    color:#f8fbff !important;
    box-shadow:none !important;
}

div[data-baseweb="select"] *{
    color:#f8fbff !important;
    -webkit-text-fill-color:#f8fbff !important;
}

div[role="listbox"], ul[role="listbox"]{
    background:#16243d !important;
    border:1px solid rgba(255,255,255,0.12) !important;
    border-radius:14px !important;
}

li[role="option"], div[role="option"]{
    color:#f8fbff !important;
    background:#16243d !important;
}
li[role="option"]:hover, div[role="option"]:hover{
    background:#22365b !important;
}
li[aria-selected="true"], div[aria-selected="true"]{
    background:#2f6df3 !important;
}

/* Vehicle profile expander */
div[data-testid="stExpander"]{
    border:none !important;
    background:transparent !important;
    box-shadow:none !important;
}
div[data-testid="stExpander"] details{
    background:linear-gradient(180deg, rgba(21,36,68,0.94), rgba(28,47,87,0.94)) !important;
    border:1px solid rgba(88,136,255,0.22) !important;
    border-radius:18px !important;
    overflow:hidden !important;
    box-shadow:0 10px 28px rgba(0,0,0,0.18) !important;
}
div[data-testid="stExpander"] summary{
    padding:16px 18px !important;
    font-size:16px !important;
    font-weight:800 !important;
    color:#f8fbff !important;
    background:rgba(255,255,255,0.02) !important;
}
div[data-testid="stExpander"] details[open] summary{
    border-bottom:1px solid rgba(255,255,255,0.08) !important;
}
div[data-testid="stExpander"] details > div{
    padding:18px 18px 14px 18px !important;
}

/* Custom stat chips */
.ds-stat-wrap{
    display:flex;
    justify-content:space-between;
    align-items:center;
    gap:12px;
    margin-top:10px;
    padding:14px 16px;
    background:rgba(255,255,255,0.04);
    border:1px solid rgba(255,255,255,0.08);
    border-radius:16px;
}
.ds-stat-grid{
    display:flex;
    gap:26px;
    flex-wrap:wrap;
}
.ds-stat-label{
    font-size:10px;
    color:#8ca6d7;
    font-family:'IBM Plex Mono', monospace;
    text-transform:uppercase;
    letter-spacing:1.4px;
    margin-bottom:5px;
}
.ds-stat-value{
    font-size:18px;
    font-weight:800;
    color:#f8fbff;
}
.ds-risk-high, .ds-risk-medium, .ds-risk-low{
    display:inline-flex;
    align-items:center;
    justify-content:center;
    min-width:128px;
    padding:10px 16px;
    border-radius:999px;
    font-family:'IBM Plex Mono', monospace;
    font-weight:800;
    font-size:13px;
    letter-spacing:1px;
}
.ds-risk-high{
    background:rgba(239,68,68,0.10);
    color:#ff6b6b;
    border:1px solid rgba(239,68,68,0.35);
}
.ds-risk-medium{
    background:rgba(245,158,11,0.10);
    color:#fbbf24;
    border:1px solid rgba(245,158,11,0.35);
}
.ds-risk-low{
    background:rgba(16,185,129,0.10);
    color:#34d399;
    border:1px solid rgba(16,185,129,0.35);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{
    gap:10px !important;
    background:transparent !important;
    border-bottom:1px solid rgba(255,255,255,0.08) !important;
}
.stTabs [data-baseweb="tab"]{
    background:rgba(255,255,255,0.04) !important;
    border:1px solid rgba(255,255,255,0.10) !important;
    border-radius:14px 14px 0 0 !important;
    padding:12px 28px !important;
    color:#dbe7ff !important;
    font-weight:800 !important;
}
.stTabs [aria-selected="true"]{
    background:#2f6df3 !important;
    border-color:#2f6df3 !important;
    color:#fff !important;
}

/* File uploader */
div[data-testid="stFileUploader"]{
    background:transparent !important;
    border:none !important;
    padding:0 !important;
}
div[data-testid="stFileUploaderDropzone"]{
    background:linear-gradient(180deg, rgba(16,27,52,0.94), rgba(25,42,78,0.92)) !important;
    border:1.5px dashed rgba(74,132,255,0.45) !important;
    border-radius:18px !important;
    min-height:150px !important;
    padding:18px !important;
}
div[data-testid="stFileUploaderDropzone"] *{
    color:#dce8ff !important;
}
div[data-testid="stFileUploader"] button{
    background:#2f6df3 !important;
    color:#fff !important;
    border:none !important;
    border-radius:12px !important;
    font-weight:800 !important;
}
div[data-testid="stFileUploader"] button *{
    color:#fff !important;
    fill:#fff !important;
}

/* Buttons */
.stButton > button{
    background:#2563eb !important;
    color:#fff !important;
    border:none !important;
    border-radius:14px !important;
    font-weight:800 !important;
    font-size:16px !important;
    padding:0.82rem 1.35rem !important;
    width:100% !important;
    box-shadow:0 8px 18px rgba(37,99,235,0.28) !important;
}
.stButton > button:hover{
    background:#1d4ed8 !important;
}

/* Result cards */
.ds-card{
    background:linear-gradient(180deg, rgba(20,34,63,0.96), rgba(28,46,84,0.95));
    border:1px solid rgba(255,255,255,0.08);
    border-radius:18px;
    padding:18px 18px;
    margin-bottom:14px;
}
.ds-card-title{
    font-size:13px;
    color:#9fb3d9;
    font-family:'IBM Plex Mono', monospace;
    text-transform:uppercase;
    letter-spacing:1.4px;
    margin-bottom:10px;
}
.ds-primary{
    font-size:26px;
    font-weight:800;
    color:#f8fbff;
    margin-bottom:6px;
}
.ds-muted{
    color:#8ea6d6;
    font-size:14px;
    font-weight:600;
}
.ds-pred-chip{
    display:inline-block;
    padding:10px 14px;
    border-radius:12px;
    background:rgba(255,255,255,0.04);
    border:1px solid rgba(255,255,255,0.08);
    margin:0 8px 8px 0;
    font-weight:700;
    color:#e4eeff;
}

/* Audio player compact */
audio{
    width:100%;
    border-radius:12px;
}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 7. LOAD MODEL
# ==============================================================================
ensure_dirs()
model, encoder = load_model_and_encoder()


# ==============================================================================
# 8. SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown(f"""
    <div class="ds-logo-wrap">
        {sound_icon_svg(50)}
        <div>
            <div class="ds-logo-name">Drive<span style="color:#3b82f6">Sense</span> AI</div>
            <div class="ds-logo-sub">Sound Diagnosis</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# 9. HEADER
# ==============================================================================
st.markdown(f"""
<div class="ds-header-wrap">
    {sound_icon_svg(78)}
    <div>
        <div class="ds-page-title">Drive<span style="color:#3b82f6">Sense</span> AI</div>
        <div class="ds-page-subtitle">AI-Enabled Car Diagnosis</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# 10. VEHICLE PROFILE
# ==============================================================================
with st.expander("🚘  Vehicle Profile — tap to set make, model & year", expanded=False):
    sorted_makes = sorted(car_data.keys())
    default_make_index = sorted_makes.index("Lexus") if "Lexus" in sorted_makes else 0

    col1, col2 = st.columns(2)
    with col1:
        v_make = st.selectbox("Make", sorted_makes, index=default_make_index, key="vp_make")

    with col2:
        default_model_index = 0
        if sorted_makes[default_make_index] == "Lexus" and "ES350" in car_data["Lexus"]:
            default_model_index = car_data["Lexus"].index("ES350")
        v_model = st.selectbox("Model", car_data[v_make], index=min(default_model_index, len(car_data[v_make]) - 1), key="vp_model")

    col3, col4 = st.columns(2)
    with col3:
        default_year_index = year_options.index(2008) if 2008 in year_options else 0
        v_year = st.selectbox("Year", year_options, index=default_year_index, key="vp_year")

    with col4:
        v_miles = st.select_slider(
            "Mileage",
            options=list(range(0, 250001, 5000)),
            value=160000,
            key="vp_mileage"
        )

    age = 2026 - v_year
    if age >= 15 or v_miles >= 150000:
        risk_class, risk_label = "ds-risk-high", "HIGH RISK"
    elif age >= 8 or v_miles >= 80000:
        risk_class, risk_label = "ds-risk-medium", "MODERATE RISK"
    else:
        risk_class, risk_label = "ds-risk-low", "LOW RISK"

    st.markdown(f"""
    <div class="ds-stat-wrap">
        <div class="ds-stat-grid">
            <div>
                <div class="ds-stat-label">Age</div>
                <div class="ds-stat-value">{age} yrs</div>
            </div>
            <div>
                <div class="ds-stat-label">Mileage</div>
                <div class="ds-stat-value">{v_miles:,} mi</div>
            </div>
            <div>
                <div class="ds-stat-label">Vehicle</div>
                <div class="ds-stat-value">{v_make} {v_model}</div>
            </div>
        </div>
        <span class="{risk_class}">{risk_label}</span>
    </div>
    """, unsafe_allow_html=True)


st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)


# ==============================================================================
# 11. INPUT STAGE
# ==============================================================================
if st.session_state.stage == "input":
    st.markdown('<div class="ds-step-badge">•&nbsp; Step 1 of 3 &nbsp;—&nbsp; Acoustic Capture</div>', unsafe_allow_html=True)
    st.markdown('<div class="ds-section">Capture your car sound</div>', unsafe_allow_html=True)
    st.markdown('<div class="ds-section-sub">Record live using your phone microphone or upload an existing audio file.</div>', unsafe_allow_html=True)

    tab_record, tab_upload = st.tabs(["Record Live", "Upload File"])
    audio_data = None

    with tab_record:
        col_l, col_c, col_r = st.columns([1, 2, 1])
        with col_c:
            st.markdown("""
            <div style="text-align:center;padding:16px 0 8px;">
                <div style="font-size:22px;font-weight:800;color:#f1f5f9;margin-bottom:4px;">Tap to record</div>
                <div style="font-size:11px;color:#7f97c3;font-family:'IBM Plex Mono',monospace;letter-spacing:1.5px;text-transform:uppercase;">
                    Hold phone near car sound
                </div>
            </div>
            """, unsafe_allow_html=True)
            recorded_audio = st.audio_input("Record", label_visibility="collapsed")

        if recorded_audio is not None:
            audio_data = recorded_audio
            st.markdown("""
            <div style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.25);
                        border-radius:12px;padding:10px 16px;margin:8px 0;text-align:center;
                        color:#34d399;font-size:14px;font-weight:700;font-family:'IBM Plex Mono',monospace;">
                ✓ Recording captured — tap Run Diagnostic Scan below
            </div>
            """, unsafe_allow_html=True)

            temp_path = audio_to_temp_file(recorded_audio, "live_recording.wav")
            st.session_state.uploaded_temp_path = temp_path
            st.session_state.uploaded_filename = "live_recording.wav"

    with tab_upload:
        uploaded_audio = st.file_uploader(
            "Upload audio",
            type=["wav", "mp3", "m4a", "aac", "flac"],
            label_visibility="collapsed"
        )

        if uploaded_audio is not None:
            audio_data = uploaded_audio
            st.audio(uploaded_audio)
            temp_path = audio_to_temp_file(uploaded_audio)
            st.session_state.uploaded_temp_path = temp_path
            st.session_state.uploaded_filename = uploaded_audio.name

            st.markdown(f"""
            <div style="background:rgba(59,130,246,0.08);border:1px solid rgba(59,130,246,0.20);
                        border-radius:12px;padding:10px 16px;margin:10px 0;text-align:center;
                        color:#93c5fd;font-size:14px;font-weight:700;font-family:'IBM Plex Mono',monospace;">
                Uploaded: {uploaded_audio.name}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    if st.button("Run Diagnostic Scan"):
        if st.session_state.uploaded_temp_path is None:
            st.warning("Please record or upload an audio file first.")
        else:
            try:
                result = predict_audio(st.session_state.uploaded_temp_path, model, encoder)
                st.session_state.result = result
                st.session_state.stage = "review"
                st.rerun()
            except Exception as e:
                st.error(f"Prediction failed: {e}")


# ==============================================================================
# 12. REVIEW STAGE
# ==============================================================================
elif st.session_state.stage == "review":
    result = st.session_state.result
    top_preds = result["top_preds"]

    st.markdown('<div class="ds-step-badge">•&nbsp; Step 2 of 3 &nbsp;—&nbsp; Review Top Predictions</div>', unsafe_allow_html=True)
    st.markdown('<div class="ds-section">Top 3 predicted sound matches</div>', unsafe_allow_html=True)
    st.markdown('<div class="ds-section-sub">Choose the reference class that sounds closest to your uploaded car noise.</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="ds-card">
        <div class="ds-card-title">Primary machine prediction</div>
        <div class="ds-primary">{result["primary_display"]}</div>
        <div class="ds-muted">Related system: {result["primary_component"]}</div>
    </div>
    """, unsafe_allow_html=True)

    chips = "".join([
        f'<span class="ds-pred-chip">{p["display"]} · {p["component"]}</span>'
        for p in top_preds
    ])
    st.markdown(f"""
    <div class="ds-card">
        <div class="ds-card-title">Top 3 predicted classes</div>
        {chips}
    </div>
    """, unsafe_allow_html=True)

    label_options = [p["display"] for p in top_preds]
    display_to_raw = {p["display"]: p["label"] for p in top_preds}

    selected_display = st.radio(
        "Choose the closest reference class",
        label_options,
        horizontal=True
    )
    selected_class = display_to_raw[selected_display]
    st.session_state.selected_reference_class = selected_class

    ref_clips = get_reference_clips_for_class(selected_class)

    st.markdown("""
    <div class="ds-card">
        <div class="ds-card-title">Reference clips</div>
        <div class="ds-muted">Play a few clips below and tick the ones that sound closest to your noise.</div>
    </div>
    """, unsafe_allow_html=True)

    selected_clip_names = []
    if ref_clips:
        for idx, clip_path in enumerate(ref_clips, start=1):
            st.markdown(f"**Reference {idx}:** `{clip_path.name}`")
            with open(clip_path, "rb") as f:
                st.audio(f.read(), format="audio/wav")
            checked = st.checkbox(f"Select {clip_path.name}", key=f"clip_{selected_class}_{idx}")
            if checked:
                selected_clip_names.append(clip_path.name)
    else:
        st.info(f"No reference clips found yet for class: {selected_class}")

    st.session_state.selected_reference_clips = selected_clip_names

    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("Back"):
            st.session_state.stage = "input"
            st.rerun()

    with col_next:
        if st.button("Confirm Closest Match"):
            saved_path = save_feedback(
                uploaded_temp_path=st.session_state.uploaded_temp_path,
                selected_class=selected_class,
                selected_clips=selected_clip_names,
                vehicle_make=v_make,
                vehicle_model=v_model,
                vehicle_year=v_year,
                mileage=v_miles,
            )
            st.session_state.saved_feedback_path = saved_path
            st.session_state.llm_report = generate_llm_summary(
                selected_class=selected_class,
                top_preds=top_preds,
                vehicle_make=v_make,
                vehicle_model=v_model,
                vehicle_year=v_year,
                mileage=v_miles,
            )
            st.session_state.stage = "report"
            st.rerun()


# ==============================================================================
# 13. REPORT STAGE
# ==============================================================================
elif st.session_state.stage == "report":
    result = st.session_state.result
    selected_class = st.session_state.selected_reference_class or result["primary_label"]
    llm_report = st.session_state.llm_report or "No report generated."

    st.markdown('<div class="ds-step-badge">•&nbsp; Step 3 of 3 &nbsp;—&nbsp; Diagnostic Insight</div>', unsafe_allow_html=True)
    st.markdown('<div class="ds-section">Diagnostic summary</div>', unsafe_allow_html=True)
    st.markdown('<div class="ds-section-sub">Final output after model prediction and user-confirmed closest reference class.</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="ds-card">
        <div class="ds-card-title">Confirmed closest sound class</div>
        <div class="ds-primary">{pretty_label(selected_class)}</div>
        <div class="ds-muted">System focus: {COMPONENT_MAP.get(selected_class, "Vehicle system")}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="ds-card">
        <div class="ds-card-title">Vehicle context</div>
        <div class="ds-muted">{v_year} {v_make} {v_model} · {v_miles:,} miles</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="ds-card">
        <div class="ds-card-title">AI diagnostic note</div>
        <div class="ds-muted" style="white-space:pre-wrap; line-height:1.7;">{llm_report}</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.saved_feedback_path:
        st.success(f"Feedback saved for later retraining: {st.session_state.saved_feedback_path.name}")

    col_restart, col_keep = st.columns(2)
    with col_restart:
        if st.button("Start New Scan"):
            st.session_state.stage = "input"
            st.session_state.result = None
            st.session_state.selected_reference_class = None
            st.session_state.selected_reference_clips = []
            st.session_state.saved_feedback_path = None
            st.session_state.llm_report = None
            st.session_state.uploaded_temp_path = None
            st.session_state.uploaded_filename = None
            st.rerun()

    with col_keep:
        if st.button("Keep This Result"):
            st.info("Result kept on screen.")