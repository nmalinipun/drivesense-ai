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
st.set_page_config(page_title="DriveSense AI", page_icon="🚗", layout="wide", initial_sidebar_state="collapsed")

# ==============================================================================
# 2. GLOBAL CONFIG
# ==============================================================================
TARGET_SR = 22050
WINDOW_SECONDS = 3.0
WINDOW_SAMPLES = int(TARGET_SR * WINDOW_SECONDS)
CONFIDENCE_THRESHOLD = 0.60

BASE_DIR         = Path(__file__).resolve().parent
MODEL_PATH       = BASE_DIR / "project_milo_final_classifier.joblib"
ENCODER_PATH     = BASE_DIR / "project_milo_label_encoder.joblib"
REFERENCE_ROOT   = BASE_DIR / "reference_audio"
FEEDBACK_ROOT    = BASE_DIR / "feedback_audio"
FEEDBACK_LOG_CSV = BASE_DIR / "feedback_log_project_milo.csv"
TEMP_UPLOAD_DIR  = BASE_DIR / "temp_uploads"

EXPECTED_CLASSES = [
    "bad_brakes","bad_ignition","low_engine_oil","low_oil_power_steering",
    "low_oil_power_steering_serpentine_belt","low_oil_serpentine_belt",
    "normal_brakes","normal_engine_idle","normal_engine_start",
    "power_steering_idle","power_steering_serpentine_belt","serpentine_belt_idle",
]

DISPLAY_NAMES = {
    "bad_brakes":                             "Bad Brake Sound",
    "bad_ignition":                           "Bad Ignition Sound",
    "low_engine_oil":                         "Low Engine Oil",
    "low_oil_power_steering":                 "Low Oil / Power Steering",
    "low_oil_power_steering_serpentine_belt": "Low Oil / Power Steering / Serpentine Belt",
    "low_oil_serpentine_belt":                "Low Oil / Serpentine Belt",
    "normal_brakes":                          "Normal Brakes",
    "normal_engine_idle":                     "Normal Engine Idle",
    "normal_engine_start":                    "Normal Engine Start",
    "power_steering_idle":                    "Power Steering at Idle",
    "power_steering_serpentine_belt":         "Power Steering / Serpentine Belt",
    "serpentine_belt_idle":                   "Serpentine Belt at Idle",
}

def pretty_label(label):
    return DISPLAY_NAMES.get(label, label.replace("_", " ").title())

# ==============================================================================
# LOGO - embedded base64 PNG
# ==============================================================================
LOGO_B64 = "iVBORw0KGgoAAAANSUhEUgAAAZAAAAGQCAYAAACAvzbMAAANyElEQVR4nO3d23kcNxKAUWg/h2CFICcmh2UlZoUg5yA/eGlTFMmZqcalCjjn1bsy3UDXL/SM1K0BAAAAAADs58PqHwBm+fT52/dZ/66vXz66t9ieTU5pM6MwithQlY1LejtEIkpcyMzmJJWTY3EvUSELG5FlxKIfUWEFm44pxGI+UWE0G4whBCMfQaE3G4puRKMOMaEHm4gwwdiHoBBh0/AQ0difmHAvG4WbRONcYsJ7bA5eJRq8JCa8ZEPwA+HgFiHhiY2AaBAmJmez+AcTDnoRkjNZ9MOIBqOJyTks9CGEg9mEZH8WeHPCwWpCsi8LuyHRICsx2YvF3IhwUIWQ7MEibkA4qEpIarN4hQkHuxCSmixaQcLBroSkFotViHBwCiGp4X+rfwDuIx6cxH6vQeWTcyNxOqeRvCxMUsIBPxKSfCxIMsIB7xOSPHwGkoh4wG3ukzyUPAE3BMQ4jazl4i8kHNCHkKzhEdYi4gH9uJ/WUO3JbHQYy2lkHieQicQDxnOfzaPUE9jQsIbTyFhOIIOJB6zj/htLQAayeWE99+E4jncD2LCQk0dafTmBdCYekJf7sy8B6cjmhPzcp/04znVgQ0JNHmld4wRykXhAXe7fawTkApsP6nMfxzm+BdhwsCePtB7jBPIg8YB9ub8fIyAPsLlgf+7z+wnInWwqOIf7/T4CcgebCc7jvr/NB0bvsIGA1ny4/hYnkDeIB/DEPHidgLzCZgFeMhd+JiAv2CTAW8yHHwnIMzYHcIs58R8B+T+bAriXefEPAWk2A/A4c0NAbAIg7PT5cXRATl984LqT58ixATl50YG+Tp0nRwbk1MUGxjlxrhwXkBMXGZjjtPlyVEBOW1xgvpPmzDEBOWlRgbVOmTdHBOSUxQTyOGHubB+QExYRyGn3+bN1QHZfPCC/nefQtgHZedGAWnadR1sGZNfFAuracS5tF5AdFwnYw27zabuAADDHVgHZre7AfnaaU9sEZKdFAfa2y7zaIiC7LAZwjh3mVvmA7LAIwJmqz6/SAal+8QEqz7HSAQFgnbIBqVxtgOeqzrOSAal6sQHeUnGulQtIxYsMcI9q861cQADIoVRAqtUZ4FGV5lyZgFS6qABXVJl3JQJS5WIC9FJh7pUICAD5pA9IhQoDjJB9/qUOSPaLBzBa5jmYNiCZLxrATFnnYdqAAJBbyoBkrS3AKhnnYrqAZLxIABlkm4/pAgJADakCkq2uANlkmpNpApLpogBklmVepgkIALWkCEiWmgJUkWFupggIAPUsD0iGigJUtHp+Lg3I6v94gOpWztHlJxAAaloWEKcPgD5WzVMnEABClgTE6QOgrxVz1QkEgJDpAXH6ABhj9nydGhDxABhr5pz1CAuAkGkBcfoAmGPWvHUCASBkSkCcPgDmmjF3nUAACBkeEKcPgDVGz18nEABChgbE6QNgrZFz+JdRvzDM8Ocfv775z377/a+JPwmc58OoX9jpg1Hei8ZbxITTff3ysfu89xkIpUTiceX/B7zNIyxK6BGAp1/DaQT6GHIC8fiKnnqfHpxGONGIuewRFqmNGvYiAtd1D4jTB0BOveezEwhpjT4lOIXANV0D4vRBL7OGu4hwmp5z2gkEgBABASCkW0A8vqKX2Y+VPMbiNL3mtRMIACFdAuL0AVBLj7ntBAJAiIAAEHI5IB5fAdR0dX47gQAQcumvc3f62Jc3/Z3Jup/n0+dv36Mvm/I+EP5175+HeP6/M1Tqs+5EeYRFa82b/k5l3bki/I5cj6/20HMQ9Ppd6YrhdNrvqDOuO2tFHmM5gRzMm/7OZN3pRUAO5U1/Z7Lu9BQKiMdXAHuJzHUnkAN509+ZrDu9CchhvOnvTNadER4OiMdXAHt6dL47gQAQIiAH8aa/M1l3RhEQAEIeCojPPwD29sicdwIBIERAAAi5OyAeXwGc4d557wQCQIgXSsEE3vTHjgQEBvGmP3Z31yMsn3/AY7zpj+rumftOINBRjwA8/RpOI2TnQ3ToxJv+OI2AQAfe9MeJPMKazLdx4Dr3UQ43A+ID9Ot8G2dvM970Zz+4j1b49Pnb969fPn546597hDWYb+PszZv+5nAf5eQR1iC+jQPXuY9ycwIZwLdx4Dr3UX4C0plv45zDm/7GcR/V8G5AfIAOcLb3OuAE0tGMb+PA7txHdQhIJ76NA9e5j2oREABCBASAkDcD4gP0+/k2DlznPsrrrR44gQAQIiAAhAgIACECAkCIgAAQIiAAhLwaEF/hBeC517rgBAJAiIAAECIgAIQICAAhAgJAiIAAECIgAIQICAAhAgJAiIAAEPJTQPw1JgC85mUfnEAACBEQAEIEBIAQAQEgREAACBEQAEIEBIAQAQEgREAACBEQAEIEBIAQAQEgREAACBEQAEIEBIAQAQEgREAACBEQAEIEBIAQAQEgREAACBEQAEIEBIAQAQEgREAACBEQAEJ+CsjXLx8/rPhBAMjtZR+cQAAIERAAQgQEgBABASBEQAAIERAAQgQEgBABASBEQAAIERAAQl4NiL/OBIDnXuuCEwgAIQICQIiAABAiIACECAgAIQICQMibAfFV3vv99vtfW//7YAb3UV5v9cAJBIAQAQEgREA6mXUcduxmZ+6jWgSko9Gb0qbnBO6jOt4NiA/SAc72XgecQDob9bsbv2vKx7eGxnEf1SAgA/TepDY9J3If5ffL6h9gV0+b9c8/fr38a8Cp3Ee5OYEMFt28Nn0NvjU0h/sop5snkK9fPn749Pnb9xk/zK6eb+L3fidls9f02+9/Xfod8j2/Pu6jFW59kcojrMlsbrjOfZSDR1jQgW8NcSIBgU58a4jTeIQFHfnWECe56wTiT6TDY3xriOrumftOIDCIbw2xOwGBCUSCHfkQHYCQuwPicxCAM9w7751AAAgREABCHgqIx1gAe3tkzjuBABAiIAfxBr0zWXdGERAAQh4OiM9BAPb06Hx3AjmMN+idybozgoAcaPRNbojkZN3pLRQQj7EA9hKZ604gh/IGvTNZd3oSkIN5g96ZrDu9XHoU9enzt++9fhDWyvQGvSs/S9SpQzDTurNO9GMJJxBaa96gdyrrzhVeKMW/vEHvTNadqMvfpvIYi948woJ5rnyr1iMsAEIuB8SfCQGo6er8dgIBIERAAAjpEhCPsQBq6TG3nUAACOkWEKcQevEGPRir17x2AgEgREAACOkaEI+x6MUb9GCMnnPaCYS0vEEPcuseEKcQgJx6z2cnEFLzBj3Ia0hAnELoyRv04LoRc9n7QCjhaeh7gx7k4REWpXiDHuQx9FGTl00xmjfowW2jPlbwCIvSRALWGfoIy4fpAGuNnMM+AwEgZHhAnEIA1hg9f51AAAiZEhCnEIC5ZsxdJxAAQqYFxCkEYI5Z89YJBICQqQFxCgEYa+acnX4CERGAMWbPV4+wAAhZEhCnEIC+VsxVJxAAQpYFxCkEoI9V89QJBICQpQFxCgG4ZuUcXX4CERGAmNXzc3lAAKgpRUBWVxSgmgxzM0VAAKgnTUAy1BSggizzMk1AWstzUQCyyjQnUwUEgDrSBSRTXQEyyTYf0wWktXwXCWC1jHMxZUAAyC9tQDLWFmCFrPMwbUBay3vRAGbJPAdTB6S13BcPYKTs8y99QADIqURAslcYoLcKc69EQFqrcTEBeqgy78oEpLU6FxUgqtKcKxUQAPIoF5BKdQZ4RLX5Vi4grdW7yAC3VJxrJQPSWs2LDfCaqvOsbEAAWKt0QKpWG+BJ5TlWOiCt1b74wNmqz6/yAWmt/iIA59lhbm0RkNb2WAzgDLvMq20C0to+iwLsa6c5tVVAAJhnu4DsVHdgL7vNp+0C0tp+iwTUt+Nc2jIgre25WEBNu86jbQPS2r6LBtSx8xzaOiCt7b14QG67z5/tA9La/osI5HPC3DkiIK2dsZhADqfMm2MC0to5iwqsc9KcOSogrZ21uMBcp82X4wLS2nmLDIx34lw5MiCtnbnYwBinzpNjA9LauYsO9HPyHDk6IK2dvfjANafPj+MD0ppNADzO3BCQf9kMwL3Mi38IyDM2BXCLOfEfAXnB5gDeYj78SEBeYZMAL5kLPxOQN9gswBPz4HUuyh0+ff72ffXPAMwnHO9zArmDTQTncd/fJiB3spngHO73+wjIA2wq2J/7/H4C8iCbC/bl/n6Mi3WBD9dhD8IR4wRygU0H9bmP4wTkIpsP6nL/XuPideSRFtQgHH04gXRkU0J+7tN+BKQzmxPycn/25WIO5JEW5CAcYziBDGTTwnruw3EEZDCbF9Zx/43l4k7kkRbMIRxzOIFMZFPDeO6zeVzoRZxGoC/hmM8JZBGbHfpxP63hoifgNAIxwrGWi5+IkMB9hCMHj7AScVPAbe6TPCxEUk4j8CPhyMeCJCcknE448rIwRQgJpxGO/HwGUoSbiZPY7zVYpIKcRtiVcNRisQoTEnYhHDVZtA0ICVUJR20WbyNCQhXCsQeLuCEhISvh2IvF3JyYsJpo7MvCHkJImE049meBDyMkjCYc57DQBxMTehGNM1l0hIQw4TibxecHYsItosETG4FXCQkvCQcv2RDcJCbnEg3eY3PwEDHZn2hwLxuFMDHZh2gQYdPQjaDUIRj0YBMxhJjkIxr0ZkMxhaDMJxiMZoOxjKj0IxasYNORiqjcJhZkYSOS3slREQsyszkpbYe4iARV2bgcY2ZsRAEAAAAAYLm/AY8fBIThWR+LAAAAAElFTkSuQmCC"

def logo_img(size=60):
    return f'<img src="data:image/png;base64,{LOGO_B64}" width="{size}px" style="border-radius:50%;display:block">'

# ==============================================================================
# 3. CUSTOM CSS
# ==============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500;600&family=Syne:wght@600;700;800&display=swap');
html,body,[class*="css"],.stApp{font-family:'Syne',sans-serif!important;background:radial-gradient(circle at top left,rgba(41,98,255,0.16),transparent 30%),radial-gradient(circle at top right,rgba(0,200,83,0.10),transparent 24%),linear-gradient(135deg,#0b1220 0%,#0f172a 45%,#172554 100%)!important;color:#f1f5f9!important}
#root > div:first-child > div > div > div > div > section > div {padding-top:0!important}
header[data-testid="stHeader"]{background:transparent!important;height:0!important;min-height:0!important}
div[data-testid="stDecoration"]{display:none!important}
div[data-testid="stToolbar"]{display:none!important}
#MainMenu{visibility:hidden!important}
footer{visibility:hidden!important}
.stApp > header{display:none!important}
.main .block-container{padding-top:1.5rem!important;padding-bottom:3rem!important;max-width:100%!important;padding-left:1.5rem!important;padding-right:1.5rem!important}
h1,h2,h3,h4,h5,h6,p,label,div,span,li,b,strong{color:#f1f5f9!important}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0b1220 0%,#0f172a 100%)!important;border-right:1px solid rgba(255,255,255,0.07)!important}
.stSelectbox label,.stSlider label,.stRadio label{font-size:13px!important;font-weight:700!important;color:#94a3b8!important;text-transform:uppercase!important;letter-spacing:1.5px!important;font-family:'IBM Plex Mono',monospace!important}
div[data-baseweb="select"]>div{background:#1e293b!important;border:1px solid rgba(255,255,255,0.12)!important;border-radius:12px!important;min-height:48px!important;color:#fff!important}
div[data-baseweb="select"] input,div[data-baseweb="select"] span,div[data-baseweb="select"] div{color:#fff!important;-webkit-text-fill-color:#fff!important}
div[data-baseweb="select"] svg{fill:#fff!important}
div[role="listbox"],ul[role="listbox"]{background:#1e293b!important;border:1px solid rgba(255,255,255,0.12)!important;border-radius:12px!important;box-shadow:0 16px 40px rgba(0,0,0,0.45)!important;padding:6px!important}
li[role="option"],div[role="option"]{background:#1e293b!important;color:#fff!important;border-radius:8px!important;margin:2px 0!important;font-size:15px!important;font-weight:600!important;padding:10px 14px!important}
li[role="option"] *,div[role="option"] *{color:#fff!important;-webkit-text-fill-color:#fff!important;opacity:1!important}
li[role="option"]:hover,div[role="option"]:hover{background:#334155!important}
li[aria-selected="true"],div[aria-selected="true"]{background:#2563eb!important}
div[data-testid="stFileUploader"]{background:transparent!important;border:none!important;padding:0!important}
div[data-testid="stFileUploaderDropzone"]{background:rgba(15,23,42,0.6)!important;border:2px dashed rgba(59,130,246,0.4)!important;border-radius:14px!important;min-height:120px!important}
div[data-testid="stAudioInput"]{background:rgba(15,23,42,0.8)!important;border:1px solid rgba(255,255,255,0.08)!important;border-radius:14px!important;padding:12px 16px!important}
div[data-testid="stFileUploaderDropzone"] *{color:#94a3b8!important;font-size:15px!important;font-weight:600!important}
div[data-testid="stFileUploader"] small{color:#64748b!important}
div[data-testid="stFileUploader"] button{background:#2563eb!important;color:#fff!important;border:none!important;border-radius:10px!important;font-weight:700!important;font-size:14px!important;padding:0.5rem 1.2rem!important}
div[data-testid="stFileUploader"] button *{color:#fff!important;fill:#fff!important}
textarea,input,.stTextArea textarea{background-color:#1e293b!important;color:#f1f5f9!important;font-size:15px!important;font-weight:600!important;border:1px solid rgba(255,255,255,0.12)!important;border-radius:12px!important}
div[data-testid="stMetric"]{background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.09);border-radius:16px;padding:20px}
div[data-testid="stMetric"] label{font-size:11px!important;font-weight:700!important;color:#64748b!important;text-transform:uppercase!important;letter-spacing:1.5px!important;font-family:'IBM Plex Mono',monospace!important}
div[data-testid="stMetricValue"]{font-size:19px!important;font-weight:800!important;color:#f1f5f9!important}
.stButton>button{background:#2563eb!important;color:#fff!important;border:none!important;border-radius:12px!important;font-weight:800!important;font-size:16px!important;padding:0.75rem 1.5rem!important;width:100%!important;font-family:'Syne',sans-serif!important;letter-spacing:0.3px!important;box-shadow:0 4px 16px rgba(37,99,235,0.3)!important}
.stButton>button:hover{background:#1d4ed8!important}
.stRadio>div{gap:8px!important}
.stRadio>div>label{background:rgba(255,255,255,0.04)!important;border:1px solid rgba(255,255,255,0.10)!important;border-radius:12px!important;padding:12px 16px!important;font-size:15px!important;font-weight:700!important;color:#f1f5f9!important;cursor:pointer!important;width:100%!important}
.stRadio>div>label:hover{border-color:#3b82f6!important;background:rgba(59,130,246,0.08)!important}
.stCheckbox label{font-size:15px!important;font-weight:700!important;color:#f1f5f9!important}
.stExpander{background:rgba(37,99,235,0.08)!important;border:1.5px solid rgba(59,130,246,0.3)!important;border-radius:16px!important;box-shadow:0 4px 16px rgba(37,99,235,0.12)!important}
.stExpander summary{font-size:15px!important;font-weight:800!important;color:#93c5fd!important;font-family:'Syne',sans-serif!important;padding:4px 0!important}
.stExpander summary:hover{color:#3b82f6!important}
/* Styled tabs */
.stTabs [data-baseweb="tab-list"]{gap:12px!important;background:transparent!important;border-bottom:2px solid rgba(255,255,255,0.06)!important;padding-bottom:0!important}
.stTabs [data-baseweb="tab"]{background:rgba(255,255,255,0.04)!important;border:1.5px solid rgba(255,255,255,0.10)!important;border-radius:12px 12px 0 0!important;color:#94a3b8!important;font-size:15px!important;font-weight:700!important;padding:12px 24px!important;font-family:'Syne',sans-serif!important}
.stTabs [aria-selected="true"]{background:#2563eb!important;border-color:#2563eb!important;color:#ffffff!important;box-shadow:0 4px 16px rgba(37,99,235,0.4)!important}
.stTabs [data-baseweb="tab"]:hover{background:rgba(59,130,246,0.12)!important;color:#f1f5f9!important}
.stTabs [data-baseweb="tab-panel"]{background:rgba(255,255,255,0.02)!important;border:1px solid rgba(255,255,255,0.06)!important;border-radius:0 16px 16px 16px!important;padding:20px!important;margin-top:-1px!important}
audio{width:100%;border-radius:10px;margin:6px 0 10px}
div[data-testid="stAlert"]{border-radius:12px!important;font-size:15px!important;font-weight:600!important}
.ds-logo-wrap{display:flex;align-items:center;gap:14px;padding:4px 0 18px;border-bottom:1px solid rgba(255,255,255,0.07);margin-bottom:18px}
.ds-logo-name{font-size:21px;font-weight:800;line-height:1;font-family:'Syne',sans-serif;margin-bottom:3px}
.ds-logo-sub{font-size:10px;font-family:'IBM Plex Mono',monospace;color:#334155!important;letter-spacing:2px;text-transform:uppercase}
.ds-page-title{font-size:2.8rem;font-weight:800;color:#f1f5f9!important;line-height:1.1;margin-bottom:4px;font-family:'Syne',sans-serif}
.ds-page-subtitle{font-size:15px;font-weight:600;color:#475569!important;font-family:'IBM Plex Mono',monospace;letter-spacing:1px;margin-bottom:1.5rem}
.ds-step-badge{display:inline-flex;align-items:center;gap:8px;font-size:11px;font-weight:700;font-family:'IBM Plex Mono',monospace;color:#60a5fa!important;background:rgba(59,130,246,0.10);border:1px solid rgba(59,130,246,0.22);border-radius:20px;padding:5px 14px;margin-bottom:10px;letter-spacing:0.5px}
.ds-section{font-size:1.7rem;font-weight:800;color:#f1f5f9!important;margin-bottom:4px;line-height:1.2;font-family:'Syne',sans-serif}
.ds-section-sub{font-size:15px;font-weight:600;color:#64748b!important;margin-bottom:1.4rem;line-height:1.6}
.ds-card{background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.09);border-radius:18px;padding:20px 24px;margin-bottom:16px;font-size:15px;font-weight:600;color:#cbd5e1!important;line-height:1.65;backdrop-filter:blur(8px)}
.ds-card b,.ds-card strong{color:#f1f5f9!important;font-weight:800!important}
.ds-soft-card{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:14px;padding:16px 20px;margin-bottom:16px}
.ds-notice{background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.25);border-left:4px solid #f59e0b;border-radius:0 14px 14px 0;padding:14px 18px;margin-bottom:20px;font-size:15px;font-weight:700;color:#fbbf24!important;line-height:1.6}
.ds-success{background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.25);border-radius:14px;padding:14px 20px;display:flex;align-items:center;gap:12px;font-size:16px;font-weight:800;color:#34d399!important;margin-bottom:20px}
.ds-pill{display:inline-block;background:rgba(59,130,246,0.12);border:1px solid rgba(59,130,246,0.25);border-radius:8px;padding:6px 14px;font-size:14px;font-weight:700;font-family:'IBM Plex Mono',monospace;color:#93c5fd!important;margin:4px 6px 4px 0}
.ds-clip-label{font-size:12px;font-weight:700;font-family:'IBM Plex Mono',monospace;color:#475569!important;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:4px}
.ds-bullet-box{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:18px;padding:4px;margin-top:8px}
.ds-bullet-item{display:flex;gap:16px;align-items:flex-start;padding:16px 18px;border-bottom:1px solid rgba(255,255,255,0.05);font-size:15px;font-weight:600;color:#cbd5e1!important;line-height:1.7}
.ds-bullet-item:last-child{border-bottom:none}
.ds-bullet-num{width:32px;height:32px;border-radius:50%;background:#2563eb;color:#fff!important;font-size:14px;font-weight:800;display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:2px;font-family:'IBM Plex Mono',monospace;box-shadow:0 2px 8px rgba(37,99,235,0.35)}
.ds-risk-high{background:rgba(239,68,68,0.15);color:#f87171!important;border:1px solid rgba(239,68,68,0.3);padding:5px 14px;border-radius:20px;font-size:12px;font-weight:800;font-family:'IBM Plex Mono',monospace;letter-spacing:1px;display:inline-block}
.ds-risk-medium{background:rgba(245,158,11,0.15);color:#fbbf24!important;border:1px solid rgba(245,158,11,0.3);padding:5px 14px;border-radius:20px;font-size:12px;font-weight:800;font-family:'IBM Plex Mono',monospace;letter-spacing:1px;display:inline-block}
.ds-risk-low{background:rgba(16,185,129,0.15);color:#34d399!important;border:1px solid rgba(16,185,129,0.3);padding:5px 14px;border-radius:20px;font-size:12px;font-weight:800;font-family:'IBM Plex Mono',monospace;letter-spacing:1px;display:inline-block}
.ds-meta{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:14px;padding:14px 20px;display:grid;grid-template-columns:repeat(3,1fr);gap:0;margin-bottom:18px}
.ds-meta-item{padding:0 16px;border-right:1px solid rgba(255,255,255,0.07)}
.ds-meta-item:first-child{padding-left:0}
.ds-meta-item:last-child{border-right:none}
.ds-meta-key{font-size:10px;font-weight:700;font-family:'IBM Plex Mono',monospace;color:#334155!important;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:5px}
.ds-meta-val{font-size:15px;font-weight:700;color:#f1f5f9!important}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 4. LOAD MODEL + ENCODER + GEMINI
# ==============================================================================
@st.cache_resource
def load_diagnostic_system():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
    if not ENCODER_PATH.exists():
        raise FileNotFoundError(f"Missing encoder file: {ENCODER_PATH}")
    model   = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    api_key = None
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", None)
    except Exception:
        api_key = None
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")
    client = None
    if api_key and genai is not None:
        try:
            client = genai.Client(api_key=api_key)
        except Exception:
            client = None
    return model, encoder, client

try:
    model, encoder, client = load_diagnostic_system()
except Exception as e:
    st.error(f"Initialization failed: {e}")
    st.stop()

loaded_classes = list(encoder.classes_)
if loaded_classes != EXPECTED_CLASSES:
    st.warning("Loaded encoder classes do not match the expected Project Milo class order.")
    st.write("Loaded classes:", loaded_classes)

# ==============================================================================
# 5. FEEDBACK HELPERS
# ==============================================================================
def ensure_feedback_dirs():
    FEEDBACK_ROOT.mkdir(parents=True, exist_ok=True)
    TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def save_uploaded_file_temporarily(uploaded_file):
    ensure_feedback_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_name = uploaded_file.name.replace(" ", "_")
    temp_path = TEMP_UPLOAD_DIR / f"{timestamp}_{safe_name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(temp_path)

def save_feedback_example(source_audio_path, selected_label, selected_clips, result, user_note):
    ensure_feedback_dirs()
    vehicle     = result["vehicle"]
    top_indices = result["top_indices"]
    selected_folder  = FEEDBACK_ROOT / selected_label
    selected_folder.mkdir(parents=True, exist_ok=True)
    timestamp        = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name        = os.path.basename(source_audio_path)
    final_audio_path = selected_folder / f"{timestamp}_{base_name}"
    shutil.copy2(source_audio_path, final_audio_path)
    top_prediction       = encoder.classes_[top_indices[0]]
    secondary_prediction = encoder.classes_[top_indices[1]] if len(top_indices) > 1 else ""
    file_exists = FEEDBACK_LOG_CSV.exists()
    with open(FEEDBACK_LOG_CSV, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["timestamp","model_family","audio_file","saved_audio_path",
                             "selected_final_class","selected_reference_clips","model_top_prediction",
                             "secondary_prediction","make","model","year","mileage",
                             "duration_sec","num_windows","user_note"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"project_milo",
                         os.path.basename(source_audio_path),str(final_audio_path),
                         selected_label,"; ".join(selected_clips),top_prediction,secondary_prediction,
                         vehicle["make"],vehicle["model"],vehicle["year"],vehicle["miles"],
                         result["duration_sec"],result["num_windows"],user_note.strip()])
    return str(final_audio_path)

# ==============================================================================
# 6. AUDIO PROCESSING
# ==============================================================================
def load_and_prepare_audio(audio_file):
    y, _ = librosa.load(str(audio_file), sr=TARGET_SR, mono=True)
    if y.size == 0:
        return y
    y_trimmed, _ = librosa.effects.trim(y, top_db=35)
    if y_trimmed.size > 0:
        y = y_trimmed
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y

def make_windows(y):
    if y.size == 0:
        return []
    if len(y) <= WINDOW_SAMPLES:
        return [np.pad(y, (0, WINDOW_SAMPLES - len(y)))]
    step    = WINDOW_SAMPLES // 2
    windows = []
    for start in range(0, len(y) - WINDOW_SAMPLES + 1, step):
        windows.append(y[start:start + WINDOW_SAMPLES])
    last_start  = len(y) - WINDOW_SAMPLES
    last_window = y[last_start:last_start + WINDOW_SAMPLES]
    if len(windows) == 0 or not np.array_equal(windows[-1], last_window):
        windows.append(last_window)
    return windows

def extract_60dim_features(y, sr):
    mfcc      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std  = np.std(mfcc,  axis=1)
    sc  = librosa.feature.spectral_centroid(y=y, sr=sr)
    sb  = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    sro = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    spectral_features = np.array([
        np.mean(sc),np.std(sc),np.mean(sb),np.std(sb),
        np.mean(sro),np.std(sro),np.mean(zcr),np.std(zcr)
    ], dtype=np.float32)
    chroma       = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean  = np.mean(chroma, axis=1)
    chroma_std   = np.std(chroma,  axis=1)
    rms          = librosa.feature.rms(y=y)
    rms_features = np.array([np.mean(rms), np.std(rms)], dtype=np.float32)
    return np.concatenate([mfcc_mean,mfcc_std,spectral_features,
                           chroma_mean,chroma_std,rms_features], axis=0).astype(np.float32).reshape(1,-1)

def _scores_to_probs(scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 0:
        scores = np.array([scores], dtype=np.float64)
    exp_scores = np.exp(scores - np.max(scores))
    return exp_scores / np.sum(exp_scores)

def predict_from_audio(audio_file):
    y = load_and_prepare_audio(audio_file)
    if y.size == 0:
        raise ValueError("Uploaded audio could not be read.")
    windows = make_windows(y)
    if not windows:
        raise ValueError("No valid audio windows could be created.")
    prob_list = []
    for win in windows:
        feats = extract_60dim_features(win, TARGET_SR)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(feats)[0]
        elif hasattr(model, "decision_function"):
            probs = _scores_to_probs(model.decision_function(feats)[0])
        else:
            raise ValueError("Model does not support probability-style inference.")
        prob_list.append(probs)
    mean_probs  = np.mean(np.vstack(prob_list), axis=0)
    top_indices = np.argsort(mean_probs)[-5:][::-1]
    all_probs   = {encoder.classes_[i]: float(mean_probs[i]) for i in range(len(encoder.classes_))}
    return {"mean_probs":mean_probs,"top_indices":top_indices,
            "num_windows":len(windows),"duration_sec":round(len(y)/TARGET_SR,2),"all_probs":all_probs}

# ==============================================================================
# 7. GEMINI HELPER
# ==============================================================================
def safe_gemini_generate(prompt):
    if client is None:
        return ""
    try:
        response = client.models.generate_content(model="gemini-3-flash-preview", contents=prompt)
        return response.text.strip()
    except Exception:
        return ""

# ==============================================================================
# 8. REFERENCE AUDIO HELPER
# ==============================================================================
def get_reference_audio_files(label):
    folder_path = REFERENCE_ROOT / label
    if not folder_path.exists() or not folder_path.is_dir():
        return []
    files = []
    for fname in sorted(os.listdir(folder_path)):
        if fname.lower().endswith((".wav", ".mp3", ".m4a")):
            files.append(str(folder_path / fname))
    return files

# ==============================================================================
# 9. EXPERT RECOMMENDATION HELPER
# ==============================================================================
def build_expert_recommendations(selected_label, vehicle, selected_clips, user_note):
    year       = vehicle["year"]
    make       = vehicle["make"]
    model_name = vehicle["model"]
    miles      = vehicle["miles"]
    high_mileage = miles >= 120000
    note_text    = (user_note or "").strip().lower()
    if selected_label == "power_steering_idle":
        bullets = [f"- Check the power steering fluid level and condition first on this {year} {make} {model_name}; low or degraded fluid can produce whining at idle.",f"- If the noise becomes stronger when turning the steering wheel, inspect the power steering pump, hoses, and steering load behavior; at {miles:,} miles, pump wear is more plausible.",f"- Also inspect the accessory belt path for slip or tension issues, because belt-related noise can overlap with steering-related whining at idle."]
    elif selected_label == "power_steering_serpentine_belt":
        bullets = [f"- Inspect both the power steering system and the serpentine belt drive, because this sound pattern overlaps between steering-load noise and belt-related squeal.",f"- Check belt condition, tension, pulley alignment, and belt glazing first on this {year} {make} {model_name}.",f"- If the sound changes when steering input is applied, also inspect power steering fluid level and pump behavior before replacing parts."]
    elif selected_label == "low_engine_oil":
        bullets = [f"- Verify engine oil level immediately and confirm oil condition; low or degraded oil can increase top-end or rotating mechanical noise.",f"- Do not continue extended driving until oil level is checked, especially on a {year} {make} {model_name} with {miles:,} miles.",f"- If oil is low, inspect for leaks, oil consumption, or overdue service rather than only topping off and moving on."]
    elif selected_label == "low_oil_power_steering":
        bullets = [f"- Check the power steering fluid reservoir level and fluid condition first; this sound pattern is consistent with steering-system fluid starvation.",f"- Inspect for hose seepage, pump-area leaks, or reservoir contamination on this {year} {make} {model_name}.",f"- If noise is strongest during steering input at idle or parking speed, treat the steering hydraulic system as the primary inspection target."]
    elif selected_label == "low_oil_power_steering_serpentine_belt":
        bullets = [f"- Inspect both fluid condition and belt-drive condition, because the selected sound class suggests overlapping steering-hydraulic and belt-drive behavior.",f"- Check power steering fluid level, pump response, belt tension, pulley condition, and visible belt glazing or cracking.",f"- Prioritize root-cause inspection before parts replacement, because low fluid and belt slip can occur together and produce similar acoustic patterns."]
    elif selected_label == "low_oil_serpentine_belt":
        bullets = [f"- Inspect engine oil condition and accessory belt condition together, because this sound class suggests overlap between lubrication-related roughness and belt-drive noise.",f"- Check for belt glazing, cracking, pulley wobble, and weak tension, especially if the sound is sharper during startup or idle transitions.",f"- Given the {miles:,} miles on this {year} {make} {model_name}, age-related wear or overdue maintenance is a realistic contributing factor."]
    elif selected_label == "bad_brakes":
        bullets = [f"- Inspect brake pad thickness, rotor surface condition, and any metallic scraping or high-pitched squeal source before continued use.",f"- If the sound occurs only during braking, prioritize front brake hardware and rotor-pad contact surfaces on this {year} {make} {model_name}.",f"- At {miles:,} miles, also check for uneven pad wear, seized slide pins, or rotor scoring rather than assuming only normal brake noise."]
    elif selected_label == "bad_ignition":
        bullets = [f"- Inspect ignition-related items first, including spark plugs, coils, and combustion smoothness, because this sound pattern is consistent with ignition irregularity.",f"- If the engine also feels rough at idle, misfires under load, or shows fuel economy drop, treat ignition diagnosis as higher priority.",f"- On a higher-mileage vehicle like this {year} {make} {model_name}, worn plugs or coil weakness are more plausible than a random isolated sound event."]
    elif selected_label == "normal_brakes":
        bullets = [f"- The selected sound is closer to a normal brake-related pattern for this {year} {make} {model_name}.",f"- Continue normal brake observation and routine service checks if braking feel, stopping distance, and pedal response remain normal.",f"- Recheck the system if the sound becomes metallic, continuous, vibration-linked, or significantly louder."]
    elif selected_label == "normal_engine_idle":
        bullets = [f"- The selected sound is closer to a normal engine idle pattern for this {year} {make} {model_name}.",f"- At {miles:,} miles, keep routine maintenance current, including oil service, belt inspection, and fluid checks, even if no active fault is indicated.",f"- Re-record and compare again if a sharper knock, whine, chirp, or rough idle develops later."]
    elif selected_label == "normal_engine_start":
        bullets = [f"- The selected sound is closer to a normal engine start pattern for this {year} {make} {model_name}.",f"- Continue standard maintenance and monitor whether startup noise becomes longer, harsher, or more metallic over time.",f"- If startup sound begins to persist after warm-up, repeat the recording and inspect lubrication and belt-drive systems."]
    elif selected_label == "serpentine_belt_idle":
        bullets = [f"- Inspect the serpentine belt first for glazing, cracking, contamination, or weak tension, since idle belt noise often points there.",f"- Also inspect belt pulleys and tensioner movement, because a worn pulley or weak tensioner can create repeating idle chirp or squeal.",f"- On a {year} {make} {model_name} with {miles:,} miles, belt-drive wear is a realistic and common inspection target."]
    else:
        bullets = [f"- Inspect the components most related to the selected sound class: {pretty_label(selected_label)}.",f"- Evaluate the sound in the context of a {year} {make} {model_name} with {miles:,} miles and compare it with the confirmed reference clips.",f"- Confirm the condition with targeted physical inspection before repair."]
    if high_mileage and selected_label not in {"normal_brakes","normal_engine_idle","normal_engine_start"}:
        bullets[1] = bullets[1].rstrip(".") + " Higher mileage increases the likelihood of wear-related causes."
    if "metallic" in note_text:
        bullets[2] = bullets[2].rstrip(".") + " A metallic character should push hard-contact components higher on the inspection list."
    return bullets[:3]

# ==============================================================================
# 10. SESSION STATE
# ==============================================================================
for key, default in {"stage":"input","result":None,"user_input":"","selected_reference_class":None,"selected_reference_clips":[],"uploaded_temp_path":None,"uploaded_filename":None,"saved_feedback_path":None}.items():
    if key not in st.session_state:
        st.session_state[key] = default

def do_restart():
    st.session_state.stage="input"; st.session_state.result=None
    st.session_state.user_input=""
    st.session_state.selected_reference_class=None
    st.session_state.selected_reference_clips=[]
    st.session_state.uploaded_temp_path=None
    st.session_state.uploaded_filename=None
    st.session_state.saved_feedback_path=None

# ==============================================================================
# 11. CAR DATA
# ==============================================================================
car_data = {"Audi":["A3","A4","A6","Q5","Q7"],"BMW":["3 Series","5 Series","X3","X5","7 Series"],"Chevrolet":["Cruze","Equinox","Malibu","Silverado","Tahoe"],"Ford":["Escape","Explorer","F-150","Focus","Fusion"],"Honda":["Accord","Civic","CR-V","Fit","Pilot"],"Hyundai":["Accent","Elantra","Santa Fe","Sonata","Tucson"],"Jeep":["Cherokee","Compass","Grand Cherokee","Renegade","Wrangler"],"Kia":["Forte","Optima","Sorento","Soul","Sportage"],"Lexus":["ES350","GX460","IS250","LS460","RX350"],"Mazda":["CX-30","CX-5","CX-9","Mazda3","Mazda6"],"Mercedes-Benz":["C-Class","E-Class","GLC","GLE","S-Class"],"Nissan":["Altima","Murano","Pathfinder","Rogue","Sentra"],"Subaru":["Crosstrek","Forester","Impreza","Legacy","Outback"],"Toyota":["Camry","Corolla","Highlander","Prius","RAV4"],"Volkswagen":["Atlas","Golf","Jetta","Passat","Tiguan"]}
year_options = list(range(2026, 1995, -1))

# ==============================================================================
# 12. SIDEBAR - logo only
# ==============================================================================
with st.sidebar:
    st.markdown(f"""
    <div class="ds-logo-wrap">
        {logo_img(46)}
        <div style="margin-left:4px">
            <div class="ds-logo-name">Drive<span style="color:#3b82f6">Sense</span>
                <span style="color:#3b82f6;font-size:14px;vertical-align:super"> AI</span>
            </div>
            <div class="ds-logo-sub">Sound Diagnosis</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 13. MAIN HEADER
# ==============================================================================
st.markdown(f"""
<div style="display:flex;align-items:center;gap:16px;margin-bottom:8px;">
    {logo_img(72)}
    <div>
        <div class="ds-page-title">Drive<span style="color:#3b82f6">Sense</span> AI</div>
        <div class="ds-page-subtitle">AI-Enabled Car Diagnosis</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Vehicle Profile — VP3 clean expander
st.markdown("""
<style>
div[data-testid="stExpander"]:first-of-type details{
    background:rgba(255,255,255,0.03)!important;
    border:1px solid rgba(255,255,255,0.08)!important;
    border-radius:14px!important;
    overflow:hidden!important;
}
div[data-testid="stExpander"]:first-of-type details summary{
    padding:14px 18px!important;
    font-size:14px!important;
    font-weight:700!important;
    color:#f1f5f9!important;
    font-family:'Syne',sans-serif!important;
    letter-spacing:0.3px!important;
}
div[data-testid="stExpander"]:first-of-type details summary:hover{
    background:rgba(255,255,255,0.04)!important;
}
</style>
""", unsafe_allow_html=True)

with st.expander("\U0001f697  Vehicle Profile  \u2014  tap to set make, model & year", expanded=False):
    sorted_makes = sorted(car_data.keys())
    default_make_index = sorted_makes.index("Lexus") if "Lexus" in sorted_makes else 0
    col1, col2 = st.columns(2)
    with col1:
        v_make = st.selectbox("Make", sorted_makes, index=default_make_index)
    with col2:
        default_model_index = 0
        if v_make == "Lexus" and "ES350" in car_data[v_make]:
            default_model_index = car_data[v_make].index("ES350")
        v_model = st.selectbox("Model", car_data[v_make], index=default_model_index)
    col3, col4 = st.columns(2)
    with col3:
        default_year_index = year_options.index(2008) if 2008 in year_options else 0
        v_year = st.selectbox("Year", year_options, index=default_year_index)
    with col4:
        v_miles = st.select_slider("Mileage", options=list(range(0, 250001, 5000)), value=160000)
    age = 2026 - v_year
    if age >= 15 or v_miles >= 150000:
        risk_class, risk_label = "ds-risk-high", "HIGH RISK"
    elif age >= 8 or v_miles >= 80000:
        risk_class, risk_label = "ds-risk-medium", "MODERATE RISK"
    else:
        risk_class, risk_label = "ds-risk-low", "LOW RISK"
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:center;
                background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
                border-radius:12px;padding:12px 16px;margin-top:8px;">
        <div style="display:flex;gap:24px;">
            <div>
                <div style="font-size:10px;color:#475569;font-family:'IBM Plex Mono',monospace;
                            text-transform:uppercase;letter-spacing:1px;">Age</div>
                <div style="font-size:15px;font-weight:700;color:#f1f5f9;">{age} yrs</div>
            </div>
            <div>
                <div style="font-size:10px;color:#475569;font-family:'IBM Plex Mono',monospace;
                            text-transform:uppercase;letter-spacing:1px;">Mileage</div>
                <div style="font-size:15px;font-weight:700;color:#f1f5f9;">{v_miles:,} mi</div>
            </div>
            <div>
                <div style="font-size:10px;color:#475569;font-family:'IBM Plex Mono',monospace;
                            text-transform:uppercase;letter-spacing:1px;">Vehicle</div>
                <div style="font-size:15px;font-weight:700;color:#f1f5f9;">{v_year} {v_make} {v_model}</div>
            </div>
        </div>
        <span class="{risk_class}">{risk_label}</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)


# ==============================================================================
# ==============================================================================
# ==============================================================================
# 14. STAGE 1: INPUT
# ==============================================================================
if st.session_state.stage == "input":
    st.markdown('<div class="ds-step-badge">&#9679;&nbsp; Step 1 of 3 &nbsp;&mdash;&nbsp; Acoustic Capture</div>', unsafe_allow_html=True)
    st.markdown('<div class="ds-section">Capture your car sound</div>', unsafe_allow_html=True)
    st.markdown('<div class="ds-section-sub">Record live using your microphone or upload an existing audio file.</div>', unsafe_allow_html=True)

    tab_record, tab_upload = st.tabs(["  Record Live", "  Upload File"])
    audio_data = None

    # ── RECORD LIVE TAB ──────────────────────────────────────────────────────
    with tab_record:
        st.markdown("""
        <div style="text-align:center;padding:16px 0 8px;">
            <div style="font-size:18px;font-weight:800;color:#f1f5f9;font-family:'Syne',sans-serif;margin-bottom:5px;">Tap to record</div>
            <div style="font-size:11px;color:#475569;font-family:'IBM Plex Mono',monospace;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:12px;">Hold phone near car sound</div>
        </div>
        """, unsafe_allow_html=True)
        recorded_audio = st.audio_input("Record car sound", label_visibility="collapsed")
        if recorded_audio is not None:
            audio_data = recorded_audio
            st.markdown('<div style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.25);border-radius:12px;padding:12px 18px;margin:8px 0;text-align:center;color:#34d399;font-size:14px;font-weight:700;font-family:IBM Plex Mono,monospace;">&#10003; Recording ready &mdash; tap Run Diagnostic Scan below</div>', unsafe_allow_html=True)
            if st.session_state.get("uploaded_filename") != "live_recording.wav":
                temp_path = save_uploaded_file_temporarily(recorded_audio)
                st.session_state.uploaded_temp_path = temp_path
                st.session_state.uploaded_filename = "live_recording.wav"
        else:
            audio_data = None

    # ── UPLOAD FILE TAB ───────────────────────────────────────────────────────
    with tab_upload:
        uploaded = st.file_uploader(
            "Drop audio file here — WAV, MP3, M4A",
            type=["wav", "mp3", "m4a"],
            label_visibility="visible"
        )
        if uploaded is not None:
            audio_data = uploaded
            st.markdown(f'<div style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.25);border-radius:12px;padding:12px 18px;margin:8px 0;display:flex;align-items:center;gap:12px;"><span style="color:#34d399;font-size:20px;">&#10003;</span><span style="color:#6ee7b7;font-size:15px;font-weight:700;">{uploaded.name}</span></div>', unsafe_allow_html=True)
            st.audio(uploaded)
            if st.session_state.get("uploaded_filename") != uploaded.name:
                temp_path = save_uploaded_file_temporarily(uploaded)
                st.session_state.uploaded_temp_path = temp_path
                st.session_state.uploaded_filename = uploaded.name

    # ── RUN SCAN BUTTON ───────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    if audio_data and st.button("Run Diagnostic Scan \u2192"):
        with st.spinner("Processing acoustic signal..."):
            try:
                if not st.session_state.uploaded_temp_path:
                    raise ValueError("Temporary uploaded file path was not created.")
                result      = predict_from_audio(st.session_state.uploaded_temp_path)
                mean_probs  = result["mean_probs"]
                top_indices = result["top_indices"]
                top_idx     = int(top_indices[0])
                top_prob    = float(mean_probs[top_idx])
                audio_name  = getattr(audio_data, "name", "live_recording.wav")
                st.session_state.result = {
                    "mean_probs":mean_probs, "top_indices":top_indices,
                    "top_idx":top_idx, "top_prob":top_prob,
                    "num_windows":result["num_windows"],
                    "duration_sec":result["duration_sec"],
                    "all_probs":result["all_probs"],
                    "audio_name":audio_name,
                    "vehicle":{"make":v_make,"model":v_model,"year":v_year,"miles":v_miles}
                }
                st.session_state.selected_reference_class = encoder.classes_[top_indices[0]]
                st.session_state.selected_reference_clips = []
                st.session_state.stage = "low_confidence" if top_prob < CONFIDENCE_THRESHOLD else "refine"
                st.rerun()
            except Exception as e:
                st.error(f"Prediction failed: {e}")

elif st.session_state.stage == "low_confidence":
    st.markdown('<div class="ds-step-badge">&#9679;&nbsp; Step 2 of 3 &nbsp;&mdash;&nbsp; Review Candidates</div>', unsafe_allow_html=True)
    st.markdown('<div class="ds-section">Review Candidate Classes</div>', unsafe_allow_html=True)
    result      = st.session_state.result
    top_indices = result["top_indices"]
    st.markdown("""<div class="ds-notice"><b>Notice:</b> The signal is not strongly separated. Review the top candidate classes below and compare them with the reference audio to confirm the correct fault category.</div>""", unsafe_allow_html=True)
    st.markdown('<div class="ds-card">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:11px;font-weight:700;color:#475569;font-family:IBM Plex Mono,monospace;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px;">Top 3 likely sound categories</div>', unsafe_allow_html=True)
    pills = "".join([f'<span class="ds-pill">{pretty_label(encoder.classes_[idx])}</span>' for idx in top_indices[:3]])
    st.markdown(pills + "</div>", unsafe_allow_html=True)
    st.markdown(f"""<div class="ds-meta"><div class="ds-meta-item"><div class="ds-meta-key">Audio file</div><div class="ds-meta-val">{result['audio_name']}</div></div><div class="ds-meta-item"><div class="ds-meta-key">Duration</div><div class="ds-meta-val">{result['duration_sec']} sec</div></div><div class="ds-meta-item"><div class="ds-meta-key">Windows</div><div class="ds-meta-val">{result['num_windows']}</div></div></div>""", unsafe_allow_html=True)
    with st.expander("Signal audit \u2014 all class probabilities"):
        st.json(result["all_probs"])
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Continue to comparison \u2192"):
            st.session_state.stage = "refine"; st.rerun()
    with col_b:
        if st.button("\u21ba Restart"):
            do_restart(); st.rerun()

# ==============================================================================
# 16. STAGE 2B: REFINE
# ==============================================================================
elif st.session_state.stage == "refine":
    st.markdown('<div class="ds-step-badge">&#9679;&nbsp; Step 2 of 3 &nbsp;&mdash;&nbsp; Collaborative Validation</div>', unsafe_allow_html=True)
    st.markdown('<div class="ds-section">Compare to Reference Audio</div>', unsafe_allow_html=True)
    st.markdown('<div class="ds-section-sub">Select the final class, then tick the reference clips that sound closest to your recording.</div>', unsafe_allow_html=True)
    result      = st.session_state.result
    top_indices = result["top_indices"]
    primary_match_raw   = encoder.classes_[top_indices[0]]
    secondary_match_raw = encoder.classes_[top_indices[1]] if len(top_indices) > 1 else "N/A"
    primary_match   = pretty_label(primary_match_raw)
    secondary_match = pretty_label(secondary_match_raw) if secondary_match_raw != "N/A" else "N/A"
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Primary Match", primary_match)
    with c2:
        st.metric("Secondary Match", secondary_match)
    st.markdown('<div class="ds-card">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:11px;font-weight:700;color:#475569;font-family:IBM Plex Mono,monospace;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:10px;">Top 5 likely sound categories</div>', unsafe_allow_html=True)
    pills = "".join([f'<span class="ds-pill">{pretty_label(encoder.classes_[idx])}</span>' for idx in top_indices])
    st.markdown(pills + "</div>", unsafe_allow_html=True)
    candidate_labels = [encoder.classes_[idx] for idx in top_indices[:4]]
    st.session_state.selected_reference_class = (st.session_state.selected_reference_class if st.session_state.selected_reference_class in candidate_labels else candidate_labels[0])
    selected_label = st.radio("Choose the final class", candidate_labels, format_func=pretty_label, index=candidate_labels.index(st.session_state.selected_reference_class))
    st.session_state.selected_reference_class = selected_label
    st.markdown(f"""<div class="ds-soft-card"><div style="font-size:11px;font-weight:700;color:#475569;font-family:'IBM Plex Mono',monospace;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:6px;">Selected final class</div><div style="font-size:20px;font-weight:800;color:#93c5fd;font-family:'IBM Plex Mono',monospace;">{pretty_label(selected_label)}</div></div>""", unsafe_allow_html=True)
    ref_files      = get_reference_audio_files(selected_label)
    selected_clips = []
    if ref_files:
        st.markdown('<div style="font-size:15px;font-weight:700;color:#94a3b8;margin-bottom:12px;">Tick the reference clips that sound closest to your car</div>', unsafe_allow_html=True)
        for i, audio_path in enumerate(ref_files, start=1):
            clip_name = os.path.basename(audio_path)
            st.markdown(f'<div class="ds-clip-label">Sample {i} &nbsp;&middot;&nbsp; {clip_name}</div>', unsafe_allow_html=True)
            st.audio(audio_path)
            checked = st.checkbox("This clip matches my car sound", key=f"clip_check_{selected_label}_{i}")
            if checked:
                selected_clips.append(clip_name)
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    else:
        st.warning(f"No reference audio files found for: {pretty_label(selected_label)}")
    st.session_state.selected_reference_clips = selected_clips
    st.markdown("<br>", unsafe_allow_html=True)
    st.session_state.user_input = st.text_area("Optional note (describe what you hear)", value=st.session_state.user_input, height=100, placeholder="Example: stronger at idle, metallic noise, sharper at startup...")
    with st.expander("Model probability audit"):
        st.json(result["all_probs"])
    st.markdown("<br>", unsafe_allow_html=True)
    col_1, col_2 = st.columns(2)
    with col_1:
        if st.button("Generate Final Report \u2192"):
            try:
                if st.session_state.uploaded_temp_path:
                    saved_path = save_feedback_example(source_audio_path=st.session_state.uploaded_temp_path,selected_label=st.session_state.selected_reference_class,selected_clips=st.session_state.selected_reference_clips,result=st.session_state.result,user_note=st.session_state.user_input)
                    st.session_state.saved_feedback_path = saved_path
            except Exception as e:
                st.warning(f"Feedback save failed: {e}")
            st.session_state.stage = "final"; st.rerun()
    with col_2:
        if st.button("\u21ba Restart"):
            do_restart(); st.rerun()

# ==============================================================================
# 17. STAGE 3: FINAL
# ==============================================================================
elif st.session_state.stage == "final":
    st.markdown('<div class="ds-step-badge">&#10003;&nbsp; Step 3 of 3 &nbsp;&mdash;&nbsp; Diagnostic Conclusion</div>', unsafe_allow_html=True)
    st.markdown('<div class="ds-section">Your Diagnosis</div>', unsafe_allow_html=True)
    result         = st.session_state.result
    vehicle        = result["vehicle"]
    selected_label = st.session_state.selected_reference_class
    selected_clips = st.session_state.selected_reference_clips
    st.markdown(f"""<div class="ds-success"><span style="font-size:22px;line-height:1;">&#10003;</span>Final signal confirmed: &nbsp;<code style="color:#6ee7b7;background:rgba(16,185,129,0.12);padding:3px 10px;border-radius:6px;font-size:15px;font-family:'IBM Plex Mono',monospace;font-weight:700;">{pretty_label(selected_label)}</code></div>""", unsafe_allow_html=True)
    st.markdown(f"""<div class="ds-card"><div style="display:grid;grid-template-columns:repeat(2,1fr);gap:14px;"><div><div style="font-size:10px;font-weight:700;color:#475569;font-family:'IBM Plex Mono',monospace;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:5px;">Vehicle</div><div style="font-size:17px;font-weight:800;color:#f1f5f9;">{vehicle['year']} {vehicle['make']} {vehicle['model']}</div></div><div><div style="font-size:10px;font-weight:700;color:#475569;font-family:'IBM Plex Mono',monospace;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:5px;">Mileage</div><div style="font-size:17px;font-weight:800;color:#f1f5f9;">{vehicle['miles']:,} mi</div></div><div><div style="font-size:10px;font-weight:700;color:#475569;font-family:'IBM Plex Mono',monospace;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:5px;">Final Signal</div><div style="font-size:17px;font-weight:800;color:#93c5fd;font-family:'IBM Plex Mono',monospace;">{pretty_label(selected_label)}</div></div><div><div style="font-size:10px;font-weight:700;color:#475569;font-family:'IBM Plex Mono',monospace;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:5px;">Reference Clips</div><div style="font-size:14px;font-weight:700;color:#64748b;font-family:'IBM Plex Mono',monospace;">{", ".join(selected_clips) if selected_clips else "None selected"}</div></div></div></div>""", unsafe_allow_html=True)
    expert_bullets = build_expert_recommendations(selected_label=selected_label,vehicle=vehicle,selected_clips=selected_clips,user_note=st.session_state.user_input)
    age_years  = 2026 - vehicle['year']
    high_miles = vehicle['miles'] >= 120000
    note_text  = (st.session_state.user_input or "").strip()
    rewrite_prompt = f"""You are a professional automotive diagnostic advisor writing a concise inspection summary for a real vehicle owner.
VEHICLE DETAILS:
- {vehicle['year']} {vehicle['make']} {vehicle['model']}
- Mileage: {vehicle['miles']:,} miles
- Age: {age_years} years old
- High mileage vehicle: {"YES" if high_miles else "NO"}
ACOUSTIC DIAGNOSIS:
- Confirmed fault class: {selected_label}
- Human-readable: {pretty_label(selected_label)}
- Reference clips matched: {", ".join(selected_clips) if selected_clips else "none"}
- Owner description: {note_text if note_text else "no additional note provided"}
EXPERT BASE RECOMMENDATIONS:
{chr(10).join(expert_bullets)}
YOUR TASK: Rewrite the 3 base recommendations into 3 clear actionable inspection steps.
Each step must: mention the {vehicle['year']} {vehicle['make']} {vehicle['model']} by name in at least one bullet, reference the mileage ({vehicle['miles']:,} mi) where relevant, be tied to "{pretty_label(selected_label)}", start with a concrete action verb.
FORMAT: Return exactly 3 lines each starting with "- ". No headers, no probabilities, no "may" or "might"."""
    bullet_text    = safe_gemini_generate(rewrite_prompt)
    gemini_bullets = [l.strip() for l in bullet_text.splitlines() if l.strip().startswith("-")]
    bullets        = gemini_bullets[:3] if len(gemini_bullets) >= 3 else expert_bullets
    st.markdown('<div style="font-size:12px;font-weight:700;color:#475569;font-family:IBM Plex Mono,monospace;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:10px;">Suggested actions</div>', unsafe_allow_html=True)
    st.markdown('<div class="ds-bullet-box">', unsafe_allow_html=True)
    for i, b in enumerate(bullets):
        text = b.lstrip("- ").strip()
        st.markdown(f'<div class="ds-bullet-item"><div class="ds-bullet-num">{i+1}</div><div>{text}</div></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    if st.session_state.saved_feedback_path:
        st.markdown(f"<p style='color:#334155;font-size:12px;font-family:IBM Plex Mono,monospace;margin-top:8px;'>Feedback saved: {st.session_state.saved_feedback_path}</p>", unsafe_allow_html=True)
    with st.expander("Computational audit"):
        st.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"Windows evaluated: {result['num_windows']}")
        st.write(f"Processed duration: {result['duration_sec']} sec")
        st.write(f"Final user-selected signal: {selected_label}")
        st.write(f"Selected reference clips: {selected_clips if selected_clips else 'None'}")
        st.write("All class probabilities:")
        st.json(result["all_probs"])
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("\u21ba Restart Diagnostic"):
        do_restart(); st.rerun()


# ==============================================================================


# ==============================================================================
# 18. CAR EXPERT CHATBOT — Self-contained HTML component
# ==============================================================================

# Build context from current session
_vehicle_ctx = ""
_diag_ctx = ""
if st.session_state.get("result"):
    _v = st.session_state.result.get("vehicle", {})
    if _v:
        _vehicle_ctx = f"{_v.get('year','')} {_v.get('make','')} {_v.get('model','')} with {_v.get('miles',0):,} miles"
    if st.session_state.get("selected_reference_class"):
        _diag_ctx = f"Diagnosis: {pretty_label(st.session_state.selected_reference_class)}"

_gemini_key = ""
try:
    _gemini_key = st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    import os
    _gemini_key = os.getenv("GEMINI_API_KEY", "")

_ctx_label = f"{_vehicle_ctx} | {_diag_ctx}" if _vehicle_ctx else "No vehicle set yet"

st.components.v1.html(f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0;font-family:'Segoe UI',sans-serif}}
body{{background:transparent}}
.fab{{position:fixed;bottom:24px;right:24px;width:56px;height:56px;border-radius:50%;
      background:#2563eb;border:none;cursor:pointer;box-shadow:0 4px 20px rgba(37,99,235,0.5);
      display:flex;align-items:center;justify-content:center;z-index:9999;transition:transform 0.2s}}
.fab:hover{{transform:scale(1.08)}}
.fab svg{{pointer-events:none}}
.panel{{position:fixed;bottom:92px;right:24px;width:360px;z-index:9998;
        background:#0f172a;border:1px solid rgba(59,130,246,0.35);
        border-radius:20px;overflow:hidden;
        display:none;flex-direction:column;
        box-shadow:0 24px 60px rgba(0,0,0,0.7);
        transition:opacity 0.2s,transform 0.2s;
        transform:translateY(10px);opacity:0}}
.panel.open{{display:flex;transform:translateY(0);opacity:1}}
.header{{background:rgba(37,99,235,0.15);border-bottom:1px solid rgba(59,130,246,0.2);
         padding:14px 16px;display:flex;align-items:center;gap:10px;flex-shrink:0}}
.avatar{{width:36px;height:36px;border-radius:50%;background:#2563eb;
         display:flex;align-items:center;justify-content:center;flex-shrink:0}}
.hname{{font-size:15px;font-weight:700;color:#f1f5f9}}
.hsub{{font-size:10px;color:#3b82f6;letter-spacing:1px;margin-top:2px;font-family:monospace}}
.dot{{width:8px;height:8px;border-radius:50%;background:#10b981;margin-left:auto;flex-shrink:0}}
.ctx-bar{{background:rgba(37,99,235,0.08);border-bottom:1px solid rgba(59,130,246,0.12);
          padding:7px 14px;font-size:11px;color:#475569;font-family:monospace;
          white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex-shrink:0}}
.ctx-bar span{{color:#3b82f6}}
.messages{{flex:1;overflow-y:auto;padding:14px;display:flex;flex-direction:column;
           gap:10px;min-height:220px;max-height:300px;scroll-behavior:smooth}}
.messages::-webkit-scrollbar{{width:4px}}
.messages::-webkit-scrollbar-track{{background:transparent}}
.messages::-webkit-scrollbar-thumb{{background:rgba(59,130,246,0.3);border-radius:2px}}
.msg-wrap.user{{display:flex;justify-content:flex-end}}
.msg-wrap.bot{{display:flex;justify-content:flex-start}}
.bubble{{padding:10px 14px;border-radius:16px;font-size:13px;line-height:1.55;
         max-width:82%;font-weight:500}}
.bubble.user{{background:#2563eb;color:#fff;border-radius:16px 16px 4px 16px}}
.bubble.bot{{background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.1);
             color:#cbd5e1;border-radius:16px 16px 16px 4px}}
.typing{{display:flex;gap:5px;align-items:center;padding:10px 14px;
         background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.08);
         border-radius:16px 16px 16px 4px;width:56px}}
.typing span{{width:6px;height:6px;border-radius:50%;background:#475569;
              animation:bounce 1.3s infinite}}
.typing span:nth-child(2){{animation-delay:0.2s}}
.typing span:nth-child(3){{animation-delay:0.4s}}
@keyframes bounce{{0%,60%,100%{{transform:translateY(0)}}30%{{transform:translateY(-5px)}}}}
.empty{{text-align:center;padding:30px 16px;color:#334155;font-size:13px;
        font-family:monospace;letter-spacing:0.5px;line-height:2}}
.chips{{padding:8px 12px;display:flex;gap:6px;flex-wrap:wrap;
        border-top:1px solid rgba(255,255,255,0.06);flex-shrink:0}}
.chip{{background:rgba(37,99,235,0.1);border:1px solid rgba(59,130,246,0.25);
       border-radius:20px;padding:5px 12px;font-size:11px;color:#93c5fd;
       font-family:monospace;cursor:pointer;transition:all 0.15s;white-space:nowrap}}
.chip:hover{{background:rgba(37,99,235,0.25);color:#fff}}
.input-row{{display:flex;gap:8px;padding:10px 12px;
            border-top:1px solid rgba(255,255,255,0.07);flex-shrink:0;align-items:center}}
.inp{{flex:1;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);
      border-radius:12px;padding:10px 14px;color:#f1f5f9;font-size:13px;outline:none;
      transition:border-color 0.2s}}
.inp:focus{{border-color:rgba(59,130,246,0.5)}}
.inp::placeholder{{color:#334155}}
.send-btn{{width:38px;height:38px;border-radius:50%;background:#2563eb;border:none;
           cursor:pointer;display:flex;align-items:center;justify-content:center;
           flex-shrink:0;transition:background 0.2s}}
.send-btn:hover{{background:#1d4ed8}}
</style>
</head>
<body>

<button class="fab" onclick="togglePanel()" title="Ask Car Expert">
  <svg width="24" height="24" viewBox="0 0 24 24" fill="white">
    <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z"/>
  </svg>
</button>

<div class="panel" id="panel">
  <div class="header">
    <div class="avatar">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="white">
        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 3c1.66 0 3 1.34 3 3s-1.34 3-3 3-3-1.34-3-3 1.34-3 3-3zm0 14.2c-2.5 0-4.71-1.28-6-3.22.03-1.99 4-3.08 6-3.08 1.99 0 5.97 1.09 6 3.08-1.29 1.94-3.5 3.22-6 3.22z"/>
      </svg>
    </div>
    <div>
      <div class="hname">Car Expert AI</div>
      <div class="hsub">POWERED BY GEMINI</div>
    </div>
    <div class="dot"></div>
  </div>
  <div class="ctx-bar"><span>&#9679;</span> {_ctx_label}</div>
  <div class="messages" id="msgs">
    <div class="empty" id="empty-msg">
      Ask me anything about your car<br>
      Engine &#183; Maintenance &#183; Costs &#183; Safety
    </div>
  </div>
  <div class="chips">
    <div class="chip" onclick="sendChip('Is it safe to drive?')">Safe to drive?</div>
    <div class="chip" onclick="sendChip('How much does an oil change cost?')">Oil change cost?</div>
    <div class="chip" onclick="sendChip('What are signs I need new brakes?')">Brake signs?</div>
    <div class="chip" onclick="sendChip('Give me a repair cost estimate')">Repair estimate</div>
  </div>
  <div class="input-row">
    <input class="inp" id="inp" placeholder="Ask anything about your car..." onkeydown="handleKey(event)"/>
    <button class="send-btn" onclick="sendMessage()">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="white">
        <path d="M2 21l21-9L2 3v7l15 2-15 2z"/>
      </svg>
    </button>
  </div>
</div>

<script>
const GEMINI_KEY = "{_gemini_key}";
const VEHICLE = "{_vehicle_ctx}";
const DIAG = "{_diag_ctx}";
let history = [];
let isOpen = false;

function togglePanel(){{
  isOpen = !isOpen;
  const p = document.getElementById('panel');
  if(isOpen){{ p.classList.add('open'); setTimeout(()=>scrollBottom(),100); }}
  else p.classList.remove('open');
}}

function scrollBottom(){{
  const m = document.getElementById('msgs');
  m.scrollTop = m.scrollHeight;
}}

function addMsg(role, text){{
  const empty = document.getElementById('empty-msg');
  if(empty) empty.remove();
  const msgs = document.getElementById('msgs');
  const wrap = document.createElement('div');
  wrap.className = 'msg-wrap ' + role;
  const bubble = document.createElement('div');
  bubble.className = 'bubble ' + role;
  bubble.textContent = text;
  wrap.appendChild(bubble);
  msgs.appendChild(wrap);
  scrollBottom();
}}

function showTyping(){{
  const msgs = document.getElementById('msgs');
  const t = document.createElement('div');
  t.id = 'typing';
  t.className = 'typing';
  t.innerHTML = '<span></span><span></span><span></span>';
  msgs.appendChild(t);
  scrollBottom();
  return t;
}}

function removeTyping(){{
  const t = document.getElementById('typing');
  if(t) t.remove();
}}

function handleKey(e){{
  if(e.key === 'Enter' && !e.shiftKey){{
    e.preventDefault();
    sendMessage();
  }}
}}

function sendChip(text){{
  document.getElementById('inp').value = text;
  sendMessage();
}}

async function sendMessage(){{
  const inp = document.getElementById('inp');
  const text = inp.value.trim();
  if(!text) return;
  inp.value = '';
  addMsg('user', text);
  history.push({{role:'user', parts:[{{text}}]}});
  const typing = showTyping();

  const ctx = (VEHICLE ? 'Vehicle: ' + VEHICLE + '. ' : '') + (DIAG ? DIAG + '. ' : '');
  const systemPrompt = 'You are an expert automotive advisor with 30 years experience. ' + ctx + 
    'Be direct, practical and concise. Max 3-4 sentences. No markdown formatting.';

  const msgs = history.map(m => ({{role: m.role, parts: m.parts}}));
  msgs.unshift({{role:'user', parts:[{{text: systemPrompt}}]}});
  msgs.splice(1,0,{{role:'model', parts:[{{text:'Understood. I am your car expert. Ask me anything.'}}]}});

  try{{
    const res = await fetch(
      'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=' + GEMINI_KEY,
      {{method:'POST', headers:{{'Content-Type':'application/json'}},
        body: JSON.stringify({{contents: history}})}}
    );
    const data = await res.json();
    removeTyping();
    const reply = data?.candidates?.[0]?.content?.parts?.[0]?.text || 'Sorry, I could not get a response.';
    addMsg('bot', reply);
    history.push({{role:'model', parts:[{{text: reply}}]}});
  }} catch(e){{
    removeTyping();
    addMsg('bot', 'Connection error. Please try again.');
  }}
}}
</script>
</body>
</html>
""", height=600, scrolling=False)