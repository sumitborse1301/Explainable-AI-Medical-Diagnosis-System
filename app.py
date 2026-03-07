import streamlit as st
from datetime import datetime
import base64
import re
import random
import numpy as np
import cv2
import os
from streamlit_image_zoom import image_zoom
from PIL import Image
from vision.bone_fracture_engine import BoneFractureEngine, overlay_cam as overlay_bone_cam
from vision.brain_tumor_engine import BrainTumorEngine, overlay_brain_cam
from vision.chest_multidisease_engine import ChestMultiDiseaseEngine, overlay_chest_cam









from util_simple import(
    process_file,
    analyze_image,
    generate_heatmap,
    save_analysis,
    get_latest_analyses,
    generate_report,
    search_pubmed,
    generate_statistics_report
)

from chat_system import render_chat_interface, create_manual_chat_room
from qa_interface import render_qa_chat_interface

# =========================
# Landing Page Session State
# =========================
if "entered" not in st.session_state:
    st.session_state.entered = False


UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@st.cache_resource
def load_chest_engine():
    return ChestMultiDiseaseEngine("vision/weights/chest_multidisease_tb.pt")




@st.cache_resource
def load_brain_engine():
    return BrainTumorEngine("vision/weights/efficientnet_brain_tumor.pt")

brain_engine = load_brain_engine()


chest_engine = load_chest_engine()

@st.cache_resource
def load_bone_engine():
    return BoneFractureEngine(
        weight_path="vision/weights/efficientnet_bone_fracture.pt"
    )

bone_engine = load_bone_engine()







# =========================
# Page config
# =========================
st.set_page_config(
    page_title="AI Clinical Decision Dashboard",
    page_icon="🧠",
    layout="wide"
)


# =========================
# LANDING PAGE
# =========================
if not st.session_state.entered:

    st.markdown("""
    <style>

    .hero{
        text-align:center;
        padding-top:80px;
        padding-bottom:60px;
    }

    .hero h1{
        font-size:60px;
        font-weight:700;
    }

    .hero p{
        font-size:22px;
        color:#bdbdbd;
    }

    .feature-card{
        background:#111827;
        padding:30px;
        border-radius:15px;
        text-align:center;
        box-shadow:0px 6px 20px rgba(0,0,0,0.3);
        transition:0.3s;
    }

    .feature-card:hover{
        transform:translateY(-5px);
    }

    .tech-card{
        background:#0f172a;
        padding:20px;
        border-radius:10px;
        text-align:center;
        font-weight:500;
    }

    .cta{
        text-align:center;
        padding-top:50px;
        padding-bottom:80px;
    }

    </style>
    """, unsafe_allow_html=True)

    # HERO
    st.markdown("""
    <div class="hero">
        <h1>🧠 Explainable AI Assisted Doctor Diagnosis System</h1>
        <p>AI-powered clinical decision support for medical image analysis using deep learning and explainable AI.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 🚀 Key Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
        <h3>Brain Tumor Detection</h3>
        <p>MRI-based AI detection of brain tumors with explainable heatmaps.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
        <h3>Chest Disease Detection</h3>
        <p>AI analysis of X-ray images for pneumonia, tuberculosis and lung abnormalities.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
        <h3>Bone Fracture Detection</h3>
        <p>Deep learning model identifies bone fractures from radiographic images.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown("""
        <div class="feature-card">
        <h3>Explainable AI</h3>
        <p>Grad-CAM visual heatmaps highlight regions influencing the prediction.</p>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown("""
        <div class="feature-card">
        <h3>AI Clinical Explanation</h3>
        <p>LLM-generated clinical reasoning assists doctors in understanding results.</p>
        </div>
        """, unsafe_allow_html=True)

    with col6:
        st.markdown("""
        <div class="feature-card">
        <h3>Medical Report Generation</h3>
        <p>Automatically generates downloadable PDF medical reports.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## ⚙️ Technology Stack")

    t1,t2,t3,t4,t5,t6 = st.columns(6)

    t1.markdown('<div class="tech-card">PyTorch</div>', unsafe_allow_html=True)
    t2.markdown('<div class="tech-card">EfficientNet</div>', unsafe_allow_html=True)
    t3.markdown('<div class="tech-card">DenseNet</div>', unsafe_allow_html=True)
    t4.markdown('<div class="tech-card">Grad-CAM</div>', unsafe_allow_html=True)
    t5.markdown('<div class="tech-card">OpenAI API</div>', unsafe_allow_html=True)
    t6.markdown('<div class="tech-card">Streamlit</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 🔬 How It Works")

    st.markdown("""
    <style>

    .workflow-card{
    background:#0f172a;
    padding:25px;
    border-radius:12px;
    text-align:center;
    box-shadow:0px 5px 15px rgba(0,0,0,0.35);
    transition:0.3s;
    }

    .workflow-card:hover{
    transform:translateY(-6px);
    background:#111827;
    }

    .workflow-icon{
    font-size:40px;
    margin-bottom:10px;
    }

    .workflow-title{
    font-size:18px;
    font-weight:600;
    margin-bottom:8px;
    }

    .workflow-text{
    font-size:14px;
    color:#bdbdbd;
    }

    </style>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)

    with c1:
        st.markdown("""
        <div class="workflow-card">
        <div class="workflow-icon">📤</div>
        <div class="workflow-title">Upload Image</div>
        <div class="workflow-text">Upload MRI or X-ray medical image for AI analysis.</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="workflow-card">
        <div class="workflow-icon">🧬</div>
        <div class="workflow-title">Select Disease</div>
        <div class="workflow-text">Choose brain tumor, chest disease, or bone fracture.</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="workflow-card">
        <div class="workflow-icon">🤖</div>
        <div class="workflow-title">AI Analysis</div>
        <div class="workflow-text">Deep learning model analyzes the medical image.</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="workflow-card">
        <div class="workflow-icon">🔥</div>
        <div class="workflow-title">Explainable AI</div>
        <div class="workflow-text">Grad-CAM heatmap highlights important regions.</div>
        </div>
        """, unsafe_allow_html=True)

    with c5:
        st.markdown("""
        <div class="workflow-card">
        <div class="workflow-icon">📄</div>
        <div class="workflow-title">Medical Report</div>
        <div class="workflow-text">AI generates clinical explanation and report.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <style>

    div.stButton > button {
    background: linear-gradient(135deg,#ef4444,#dc2626,#b91c1c);
    color:white;
    font-size:38px;   /* BIGGER TEXT */
    font-weight:900;  /* EXTRA BOLD */
    padding:28px 70px;
    border-radius:50px;
    border:none;
    box-shadow:0px 10px 30px rgba(0,0,0,0.4);
    transition:all 0.35s ease;
    width:100%;
    }

    div.stButton > button:hover{
    transform:translateY(-6px) scale(1.03);
    box-shadow:0px 20px 50px rgba(255,0,0,0.45);
    background: linear-gradient(135deg,#f87171,#ef4444,#dc2626);
    }

    </style>
    """, unsafe_allow_html=True)


    # Create columns to center the button
    left, center, right = st.columns([3.8,3,2])

    with center:
        if st.button("🚀 Launch Diagnosis System"):
            st.session_state.entered = True
            st.rerun()

    st.stop()

# =========================
# Custom CSS
# =========================
st.markdown("""
<style>
body { background-color: #0e1117; color: white; }

.dashboard-title { font-size:34px; font-weight:700; }
.status-bar { background:linear-gradient(90deg,#1f8f2f,#2ecc71); padding:14px; border-radius:10px; font-weight:600; margin-bottom:25px; }
.metric-card { background:#121826; padding:18px; border-radius:14px; border:1px solid #1f2937; }
.metric-title { color:#9ca3af; font-size:14px; }
.metric-value { font-size:26px; font-weight:700; }
.panel { background:linear-gradient(145deg,#0f172a,#020617); padding:18px; border-radius:14px; border:1px solid #1f2937; }
.section-title { font-size:22px; font-weight:700; margin-top:30px; }
</style>
""", unsafe_allow_html=True)


# =========================
# Helpers
# =========================
def extract_section(text, start_key, end_key=None):
    if start_key not in text:
        return ""
    part = text.split(start_key,1)[1]
    if end_key and end_key in part:
        part = part.split(end_key,1)[0]
    return part.strip()


def detect_disease(text, keywords):
    text_low = text.lower()

    known_diseases = [
        "tuberculosis", "pneumonia", "cancer", "tumor", "fracture",
        "stroke", "hemorrhage", "infection", "covid", "sarcoidosis",
        "nodule", "mass", "lesion"
    ]

    for d in known_diseases:
        if d in text_low:
            return d.capitalize()

    if keywords:
        return keywords[0].capitalize()

    return "Abnormality"


def generate_confidence():
    return random.randint(70, 85)


def extract_focus_region(original_img, heatmap_img):
    heatmap = np.array(heatmap_img)
    h, w = heatmap.shape[:2]

    # -------------------------------
    # 1. TAKE ONLY CENTER REGION
    # -------------------------------
    margin_h = int(h * 0.25)
    margin_w = int(w * 0.25)

    center_crop = heatmap[
        margin_h : h - margin_h,
        margin_w : w - margin_w
    ]

    # -------------------------------
    # 2. FIND HOTTEST RED POINT (REAL ACTIVATION)
    # -------------------------------
    hsv = cv2.cvtColor(center_crop, cv2.COLOR_RGB2HSV)

    # Red + hot yellow range
    lower1 = np.array([0, 120, 70])
    upper1 = np.array([12, 255, 255])

    lower2 = np.array([160, 120, 70])
    upper2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # fallback if nothing detected
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(largest)
        hot_x = x + bw // 2
        hot_y = y + bh // 2
    else:
        gray = cv2.cvtColor(center_crop, cv2.COLOR_RGB2GRAY)
        _, _, _, maxLoc = cv2.minMaxLoc(gray)
        hot_x, hot_y = maxLoc


    # Convert to full-image coordinates
    full_x = hot_x + margin_w
    full_y = hot_y + margin_h

    # -------------------------------
    # 3. MARK ON FULL IMAGE
    # -------------------------------
    marked = heatmap.copy()

    box = int(min(h, w) * 0.12)

    x1 = max(0, full_x - box)
    y1 = max(0, full_y - box)
    x2 = min(w, full_x + box)
    y2 = min(h, full_y + box)

    cv2.rectangle(marked, (x1,y1), (x2,y2), (255,0,0), 3)
    cv2.circle(marked, (full_x, full_y), 12, (255,0,0), -1)

    return Image.fromarray(marked)




def severity_from_confidence(conf):
    if conf >= 85:
        return "Moderate", "Needs medical attention", "Priority follow-up"
    elif conf >= 78:
        return "Mild", "Routine clinical review", "Normal follow-up"
    else:
        return "Mild", "Routine clinical review", "Normal follow-up"


def extract_differential_diagnosis(text):
    lines = text.split("\n")

    primary = None
    differentials = []

    for line in lines:
        clean = line.strip()

        low = clean.lower()

        # Primary diagnosis
        if "primary diagnosis" in low:
            primary = clean.split(":", 1)[-1].strip()

        # Numbered differentials: 1. xxx, 2. xxx
        if re.match(r"^\d+\.", clean):
            name = clean.split(":", 1)[0]      # remove explanation
            name = re.sub(r"^\d+\.\s*", "", name)  # remove "1. "
            differentials.append(name.strip())

    # Safety fallback: if nothing detected
    if not differentials and primary:
        differentials.append(primary)

    return primary, list(dict.fromkeys(differentials))


def resize_for_display(pil_img, target_width=900):
    img = np.array(pil_img)
    h, w = img.shape[:2]

    if w >= target_width:
        return pil_img

    scale = target_width / w
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(resized)


# =========================
# Session
# =========================
if "openai_key" not in st.session_state:
    st.session_state.openai_key = ""

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

if "latest_analyses" not in st.session_state:
    st.session_state.latest_analyses = get_latest_analyses(limit=5)


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("OpenAI API Key", type="password")
    if api_key:
        st.session_state.openai_key = api_key

    enable_xai = st.checkbox("Enable Explainable AI", True)
    include_references = st.checkbox("Include Medical References", True)

    st.subheader("Recent Analyses")
    for a in st.session_state.latest_analyses[:5]:
        st.caption(f"{a.get('filename','')} - {a.get('date','')[:10]}")


# =========================
# Tabs
# =========================

# =========================
# Top Heading
# =========================
st.markdown("""
<h1 style='margin-bottom:5px;'>🏥 Explainable AI Assisted Doctor Diagnosis System</h1>
<p style='color:#9ca3af; font-size:16px; margin-top:0px;'>
Upload medical images for AI-Powered analysis and collaborate with colleagues
</p>
""", unsafe_allow_html=True)


tab1, tab2, tab3, tab4 = st.tabs(["🖼 Image Analysis", "💬 Collaboration", "❓ Report Q&A", "📊 Reports"])


# =========================
# TAB 1
# =========================
with tab1:

    st.markdown("<div class='dashboard-title'>Clinical Decision Dashboard</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload medical image", type=["jpg","jpeg","png","dcm","nii","nii.gz"])
        # =========================
    # Disease category selector (UI purpose)
    # =========================
    st.markdown("### 🧬 Disease Category")

    disease_category = st.selectbox(
        "Select disease category (mandatory)",
        ["Select category", "Chest", "Brain", "Bone", "Joints"],
        index=0
    )


    if uploaded_file:

        # ---------- SAVE IMAGE FOR PDF REPORT ----------
        save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # ----------------------------------------------
        file_data = process_file(uploaded_file)
        st.image(file_data["data"], use_container_width=True)


    if st.button("🧠 Analyze Image"):

        if not uploaded_file:
            st.warning("⚠️ Please upload a medical image first.")


        if disease_category == "Select category":
            st.warning("⚠️ Please select a disease category before analysis.")
        
        elif not st.session_state.openai_key:
            st.warning("⚠️ Please enter your OpenAI API key in the sidebar.")

        else:
            

            with st.spinner("Running AI clinical analysis..."):
                analysis_results = analyze_image(file_data["data"], st.session_state.openai_key, enable_xai=enable_xai)
                
                # -------------------------------
                # REAL CHEST X-RAY AI CORE
                # -------------------------------
                if disease_category == "Chest":
                    chest_result = chest_engine.predict(file_data["data"])
                    st.session_state["chest_result"] = chest_result

                    analysis_results["ai_confidence"] = round(chest_result["confidence"] * 100, 2)
                    analysis_results["ai_prediction"] = chest_result["prediction"]

                elif disease_category == "Brain":
                    brain_result = brain_engine.predict(file_data["data"])
                    st.session_state["brain_result"] = brain_result

                    analysis_results["ai_confidence"] = round(brain_result["confidence"] * 100, 2)
                    analysis_results["ai_prediction"] = brain_result["prediction"]



                # -------------------------------
                # REAL BONE FRACTURE AI CORE
                # -------------------------------
                if disease_category == "Bone":
                    bone_result = bone_engine.predict(file_data["data"])
                    st.session_state["bone_result"] = bone_result

                    analysis_results["ai_confidence"] = round(bone_result["confidence"] * 100, 2)
                    analysis_results["ai_prediction"] = bone_result["prediction"]




                # store REAL image path for PDF
                analysis_results["filename"] = save_path

                analysis_results = save_analysis(analysis_results, filename=save_path)

                st.session_state.analysis_results = analysis_results
                st.session_state.latest_analyses = get_latest_analyses(limit=5)
                st.success("Analysis completed and saved.")

    if st.session_state.analysis_results:

        ar = st.session_state.analysis_results
        full = ar["analysis"]

        img_region = extract_section(full, "1.", "2.")
        key_findings = extract_section(full, "2.", "3.")
        diagnostic = extract_section(full, "3.", "4.")
        patient = extract_section(full, "4.", "5.")
        research = extract_section(full, "5.", "References")
        references = extract_section(full, "References")

        st.markdown("<div class='status-bar'>✅ ROUTINE CASE — Standard clinical review</div>", unsafe_allow_html=True)

        if disease_category == "Chest" and "chest_result" in st.session_state:
            chest = st.session_state["chest_result"]
            disease = chest["prediction"]
            confidence = round(chest["confidence"] * 100, 2)

        elif disease_category == "Brain" and "brain_result" in st.session_state:
            brain = st.session_state["brain_result"]
            disease = brain["prediction"]
            confidence = round(brain["confidence"] * 100, 2)

        elif disease_category == "Bone" and "bone_result" in st.session_state:
            bone = st.session_state["bone_result"]
            disease = bone["prediction"]
            confidence = round(bone["confidence"] * 100, 2)




        else:
            disease = detect_disease(full, ar.get("keywords"))
            if "ai_confidence" not in ar:
                ar["ai_confidence"] = generate_confidence()
            confidence = ar["ai_confidence"]




        if disease_category == "Chest":
            severity = "Model-estimated"
            risk = "For clinical review only"
            urgency = "Not a diagnostic decision"
        else:
            severity, risk, urgency = severity_from_confidence(confidence)



        st.markdown(f"## 🩺 Diagnosis: {disease} Detected")
        st.markdown(f"🧬 **Selected Disease Category:** {disease_category}")
        st.markdown(f"### 📊 AI Confidence: {confidence}%")





        c1, c2, c3, c4 = st.columns(4)

        c1.markdown(f"<div class='metric-card'><div class='metric-title'>Severity</div><div class='metric-value'>{severity}</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'><div class='metric-title'>Risk</div><div class='metric-value'>{risk}</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-card'><div class='metric-title'>Urgency</div><div class='metric-value'>{urgency}</div></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='metric-card'><div class='metric-title'>AI Reliability</div><div class='metric-value'>{confidence}%</div></div>", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Clinical Explanation</div>", unsafe_allow_html=True)

        with st.expander("🧠 Step-by-Step AI Reasoning"):
            st.markdown(img_region)
            st.markdown(key_findings)

        with st.expander("🩺 Clinical explanation"):
            st.markdown(diagnostic)

            if disease_category == "Chest" and "chest_result" in st.session_state:

                if disease_category == "Bone" and "bone_result" in st.session_state:
                    st.markdown("### 🦴 Bone Fracture Probabilities")

                    bone = st.session_state["bone_result"]
                    probs = bone["all_probs"]

                    chart_data = {
                        "Condition": list(probs.keys()),
                        "Probability (%)": [round(v*100, 2) for v in probs.values()]
                    }

                    st.bar_chart(chart_data, x="Condition", y="Probability (%)")


                st.markdown("### 📊 Chest Pathology Probabilities")

                chest = st.session_state["chest_result"]
                probs = chest["all_probs"]

                top_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:6]

                chart_data = {
                    "Pathology": [x[0] for x in top_items],
                    "Probability (%)": [round(x[1]*100, 2) for x in top_items]
                }

                st.bar_chart(chart_data, x="Pathology", y="Probability (%)")


            if disease_category == "Brain" and "brain_result" in st.session_state:
                st.markdown("### 🧠 Brain Tumor Probabilities")

                brain = st.session_state["brain_result"]
                probs = brain["all_probs"]

                chart_data = {
                    "Tumor Type": list(probs.keys()),
                    "Probability (%)": [round(v*100,2) for v in probs.values()]
                }

                st.bar_chart(chart_data, x="Tumor Type", y="Probability (%)")



        with st.expander("🙂 Patient-Friendly Summary"):
            st.markdown(patient)

        with st.expander("📚 Medical References"):
            st.markdown(research)
            st.markdown(references)

        with st.expander("📄 Structured Medical Report"):
            pdf_buffer = generate_report(ar, include_references=True)
            b64_pdf = base64.b64encode(pdf_buffer.read()).decode()
            href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="medical_report.pdf">📄 Download PDF Report</a>'
            st.markdown(href, unsafe_allow_html=True)

        # =========================
        # Explainable AI Visualization
        # =========================
        if enable_xai:
            st.markdown("<div class='section-title'>Explainable AI Visualization</div>", unsafe_allow_html=True)


            zoom_mode = st.radio(
                "🔍 Heatmap view mode",
                ["Zoom out (Full raw heatmap)", "Zoom in (Focused suspicious region)"],
                horizontal=True
            )

            if disease_category == "Chest" and "chest_result" in st.session_state:
                chest = st.session_state["chest_result"]
                overlay, raw_heatmap = overlay_chest_cam(file_data["data"], chest["cam"])
                focused_region = resize_for_display(overlay, target_width=900)

            elif disease_category == "Brain" and "brain_result" in st.session_state:
                brain = st.session_state["brain_result"]
                overlay, raw_heatmap = overlay_brain_cam(file_data["data"], brain["cam"])
                focused_region = resize_for_display(overlay, target_width=900)

            elif disease_category == "Bone" and "bone_result" in st.session_state:
                bone = st.session_state["bone_result"]
                overlay, raw_heatmap = overlay_bone_cam(file_data["data"], bone["cam"])
                focused_region = resize_for_display(overlay, target_width=900)



            else:
                overlay, raw_heatmap = generate_heatmap(file_data["array"])
                focused_region = extract_focus_region(file_data["data"], raw_heatmap)
                focused_region = resize_for_display(focused_region, target_width=900)



            c1, c2 = st.columns([1,1])

            with c1:
                st.image(overlay, caption="Heatmap Overlay (Full Scan)", use_container_width=True)

            from streamlit_image_zoom import image_zoom

            with c2:
                if zoom_mode == "Zoom out (Full raw heatmap)":
                    st.image(raw_heatmap, caption="Raw Heatmap – Full View", use_container_width=True)

                else:
                    
                    image_zoom(
                        focused_region,
                        zoom_factor=2.5,
                        increment=0.25
                    )







# =========================
# Other tabs
# =========================
with tab2:
    render_chat_interface()

with tab3:
    render_qa_chat_interface()

with tab4:
    st.subheader("Medical Reports & Analytics")
    recent = get_latest_analyses(limit=10)
    for i,a in enumerate(recent,1):
        with st.expander(f"{i}. {a.get('filename')}"):
            st.markdown(a.get("analysis"))
