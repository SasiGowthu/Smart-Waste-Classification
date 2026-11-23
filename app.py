
from pathlib import Path
from datetime import datetime
from collections import Counter
import csv
import io

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ==========================================
# CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Smart Waste AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = Path("models/waste_classifier.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# CUSTOM DARK THEME
# ==========================================

st.markdown("""
<style>
.stApp {
    background: #090f0d;
    color: #e8f1ec;
}
[data-testid="stSidebar"] {
    background: #101b16;
    border-right: 1px solid #243b30;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}
h1, h2, h3 {
    color: #e8f5ed !important;
}
p, label {
    color: #b6c9bd;
}
div[data-testid="stMetric"] {
    background: #14231b;
    border: 1px solid #294b38;
    padding: 18px;
    border-radius: 12px;
}
div[data-testid="stMetricValue"] {
    color: #4ade80;
}
.stButton > button {
    background: #16a34a;
    color: white;
    border: none;
    border-radius: 9px;
    font-weight: 600;
    min-height: 42px;
}
.stButton > button:hover {
    background: #15803d;
    color: white;
    border: 1px solid #4ade80;
}
.hero {
    background: linear-gradient(120deg, #10291d, #14251d);
    border: 1px solid #28553a;
    border-radius: 18px;
    padding: 28px;
    margin-bottom: 24px;
}
.hero-title {
    font-size: 36px;
    font-weight: 800;
    color: #4ade80;
}
.hero-subtitle {
    color: #b6c9bd;
    font-size: 15px;
}
.result-card {
    background: #14231b;
    border: 1px solid #2d6141;
    padding: 20px;
    border-radius: 14px;
}
.small-muted {
    color: #8fa99a;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# MODEL LOADING
# ==========================================

@st.cache_resource
def load_classifier():
    if not MODEL_PATH.exists():
        return None, None, None, None

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    class_names = checkpoint["class_names"]
    image_size = checkpoint.get("image_size", 224)

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(
        model.last_channel,
        len(class_names)
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    return model, class_names, preprocess, image_size


# ==========================================
# DISPOSAL RECOMMENDATIONS
# ==========================================

DISPOSAL_GUIDE = {
    "battery": (
        "Battery waste",
        "Keep batteries separate from ordinary waste. "
        "Take them to an authorised battery collection point."
    ),
    "biological": (
        "Organic waste",
        "Separate food and garden waste. Compost suitable "
        "organic materials where facilities are available."
    ),
    "brown-glass": (
        "Brown glass",
        "Keep glass separate and use an authorised glass "
        "collection or recycling facility."
    ),
    "green-glass": (
        "Green glass",
        "Separate glass from general waste. Use a local "
        "glass collection point."
    ),
    "white-glass": (
        "Clear glass",
        "Keep glass separate. Handle broken glass carefully "
        "and follow local recycling rules."
    ),
    "cardboard": (
        "Cardboard",
        "Keep cardboard clean and dry. Flatten boxes before "
        "placing them in the paper recycling stream."
    ),
    "paper": (
        "Paper",
        "Keep paper dry and clean. Separate it for paper recycling."
    ),
    "clothes": (
        "Textile waste",
        "Donate or reuse wearable clothes. Otherwise check "
        "local textile collection and recycling facilities."
    ),
    "metal": (
        "Metal waste",
        "Separate metal items and place accepted materials "
        "in the dry recycling stream."
    ),
    "plastic": (
        "Plastic waste",
        "Empty and rinse plastic containers when practical. "
        "Check whether your local collection accepts the plastic."
    ),
    "shoes": (
        "Footwear waste",
        "Donate usable shoes. Check textile or footwear "
        "collection options for damaged items."
    ),
    "trash": (
        "Residual waste",
        "Check whether the item can be reused or recycled "
        "before placing it in residual waste."
    )
}


def disposal_advice(label):
    key = label.lower().replace("_", "-")
    return DISPOSAL_GUIDE.get(
        key,
        ("General waste", "Follow your local municipal waste rules.")
    )


def display_name(label):
    return label.replace("_", " ").replace("-", " ").title()


# ==========================================
# SESSION STATE
# ==========================================

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ==========================================
# LOAD MODEL
# ==========================================

model, class_names, preprocess, image_size = load_classifier()

if model is None:
    st.error("Trained model not found.")
    st.info("Expected file: models/waste_classifier.pth")
    st.stop()


# ==========================================
# TOP NAVIGATION
# ==========================================

st.markdown("""
<div class="hero">
    <div class="hero-title">♻️ Smart Waste AI</div>
    <div class="hero-subtitle">INTELLIGENT WASTE CLASSIFICATION</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Clean button-based top navigation: no radio controls */
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button {
    width: 100%;
    min-height: 46px;
    border-radius: 11px;
    border: 1px solid #294b38;
    background: #14251d;
    color: #e5f5eb;
    font-weight: 600;
    white-space: nowrap;
    transition: all 0.2s ease;
}
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button:hover {
    border-color: #55d889;
    background: #1d3b2b;
    color: #ffffff;
}
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button[kind="primary"] {
    background: #245c3b;
    border: 1px solid #55d889;
    color: #ffffff;
    box-shadow: 0 0 10px rgba(85, 216, 137, 0.18);
}
</style>
""", unsafe_allow_html=True)

page_options = {
    "🏠 Home": "Home",
    "🔍 Classify Waste": "Classify Waste",
    "📊 Analytics": "Analytics",
    "🕒 Prediction History": "Prediction History",
    "♻️ Recycling Guide": "Recycling Guide",
    "🧠 About Model": "About Model"
}

if "active_page" not in st.session_state:
    st.session_state.active_page = "🏠 Home"

nav_columns = st.columns(len(page_options), gap="small")
for column, (label, page_name) in zip(nav_columns, page_options.items()):
    with column:
        if st.button(
            label,
            key=f"nav_{page_name.lower().replace(' ', '_')}",
            type="primary" if st.session_state.active_page == label else "secondary",
            use_container_width=True
        ):
            st.session_state.active_page = label
            st.rerun()

page = page_options[st.session_state.active_page]

st.divider()


# ==========================================
# HOME PAGE
# ==========================================

if page == "Home":

    st.markdown("""
    <div class="hero">
        <div class="hero-title">♻️ Smart Waste AI</div>
        <div class="hero-subtitle">
            Intelligent waste classification using deep learning.
            Identify waste, understand its category and discover
            responsible disposal methods.
        </div>
    </div>
    """, unsafe_allow_html=True)

    total = len(st.session_state.history)
    unique = len(set(x["category"] for x in st.session_state.history))

    c1, c2, c3 = st.columns(3)

    c1.metric("Waste Categories", len(class_names))
    c2.metric("Images Classified", total)
    c3.metric("Categories Detected", unique)

    st.markdown("## Explore Smart Waste AI")

    a, b = st.columns(2)

    with a:
        st.markdown("### 🔍 Waste Classification")
        st.write("Upload or capture a waste image and get AI predictions.")

        if st.button("Open Classification", use_container_width=True):
            st.session_state.navigation_target = "Classify Waste"
            st.rerun()

    with b:
        st.markdown("### 📊 Analytics Dashboard")
        st.write("Review your classification activity and category statistics.")

        if st.button("Open Analytics", use_container_width=True):
            st.session_state.navigation_target = "Analytics"
            st.rerun()

    st.markdown("## Supported Waste Categories")

    cols = st.columns(4)

    for i, name in enumerate(class_names):
        with cols[i % 4]:
            st.markdown(f"**♻️ {display_name(name)}**")


# ==========================================
# CLASSIFICATION PAGE
# ==========================================

elif page == "Classify Waste":

    st.title("🔍 Waste Classification")
    st.write("Upload an image or use your camera to identify waste.")

    upload_col, camera_col = st.columns(2)

    with upload_col:
        uploaded = st.file_uploader(
            "Upload waste image",
            type=["jpg", "jpeg", "png"]
        )

    with camera_col:
        camera_image = st.camera_input("Capture using camera")

    source = uploaded if uploaded is not None else camera_image

    if source is not None:

        try:
            image = Image.open(source).convert("RGB")

            left, right = st.columns([1, 1])

            with left:
                st.image(
                    image,
                    caption="Selected waste image",
                    use_container_width=True
                )

            with right:
                st.markdown("### Ready to classify")
                st.write("Your image is ready for AI analysis.")

                if st.button("♻️ Classify Waste", use_container_width=True):

                    with st.spinner("Analysing image..."):

                        input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

                        with torch.inference_mode():
                            logits = model(input_tensor)
                            probabilities = torch.softmax(logits, dim=1)[0]

                        top_probs, top_indices = torch.topk(
                            probabilities,
                            min(3, len(class_names))
                        )

                        predictions = []

                        for prob, idx in zip(
                            top_probs.tolist(),
                            top_indices.tolist()
                        ):
                            predictions.append({
                                "category": class_names[idx],
                                "confidence": round(prob * 100, 2)
                            })

                        best = predictions[0]
                        advice_title, advice_text = disposal_advice(
                            best["category"]
                        )

                        result = {
                            "category": best["category"],
                            "confidence": best["confidence"],
                            "predictions": predictions,
                            "advice_title": advice_title,
                            "advice": advice_text,
                            "time": datetime.now().strftime(
                                "%d %b %Y, %I:%M:%S %p"
                            )
                        }

                        st.session_state.last_result = result
                        st.session_state.history.append(result)

            result = st.session_state.last_result

            if result is not None:

                st.divider()
                st.markdown("## 🎯 Classification Results")

                r1, r2 = st.columns(2)

                with r1:
                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div class="small-muted">PREDICTED CATEGORY</div>
                            <h2>{display_name(result["category"])}</h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with r2:
                    st.metric(
                        "Prediction confidence",
                        f'{result["confidence"]:.2f}%'
                    )

                st.markdown("### Top 3 Predictions")

                for prediction in result["predictions"]:
                    st.write(
                        f'**{display_name(prediction["category"])}**'
                    )
                    st.progress(
                        prediction["confidence"] / 100,
                        text=f'{prediction["confidence"]:.2f}%'
                    )

                st.markdown("### ♻️ Disposal Recommendation")
                st.success(result["advice_title"])
                st.write(result["advice"])

                st.caption(
                    "Predictions are model estimates, not guaranteed "
                    "identifications. Follow local waste regulations."
                )

        except Exception as error:
            st.error(f"Unable to process this image: {error}")


# ==========================================
# ANALYTICS PAGE
# ==========================================

elif page == "Analytics":

    st.title("📊 Analytics Dashboard")
    st.write("Overview of your waste classification activity.")

    history = st.session_state.history

    total = len(history)

    if total == 0:
        st.info("No predictions yet. Classify some images to see analytics.")

    else:
        counts = Counter(x["category"] for x in history)
        avg_confidence = sum(
            x["confidence"] for x in history
        ) / total

        c1, c2, c3 = st.columns(3)

        c1.metric("Total Predictions", total)
        c2.metric("Categories Detected", len(counts))
        c3.metric("Average Confidence", f"{avg_confidence:.2f}%")

        st.markdown("## Category Distribution")

        chart_data = {
            display_name(k): v for k, v in counts.items()
        }

        st.bar_chart(chart_data)

        st.markdown("## Prediction Details")

        for category, count in counts.most_common():
            st.write(f"**{display_name(category)}:** {count} predictions")


# ==========================================
# PREDICTION HISTORY
# ==========================================

elif page == "Prediction History":

    st.title("🕒 Prediction History")
    st.write("Previously classified images in this session.")

    history = st.session_state.history

    if not history:
        st.info("Your prediction history is currently empty.")

    else:
        for index, item in enumerate(reversed(history), start=1):

            with st.expander(
                f'{display_name(item["category"])} — '
                f'{item["confidence"]:.2f}%'
            ):
                st.write(f'**Time:** {item["time"]}')
                st.write(f'**Category:** {display_name(item["category"])}')
                st.write(f'**Confidence:** {item["confidence"]:.2f}%')
                st.write(f'**Advice:** {item["advice"]}')

        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow([
            "Time",
            "Category",
            "Confidence",
            "Disposal Recommendation"
        ])

        for item in history:
            writer.writerow([
                item["time"],
                item["category"],
                item["confidence"],
                item["advice"]
            ])

        st.download_button(
            "⬇️ Download History CSV",
            output.getvalue(),
            file_name="waste_prediction_history.csv",
            mime="text/csv"
        )

        if st.button("Clear Prediction History"):
            st.session_state.history = []
            st.session_state.last_result = None
            st.rerun()


# ==========================================
# RECYCLING GUIDE
# ==========================================

elif page == "Recycling Guide":

    st.title("♻️ Recycling & Disposal Guide")
    st.write("Explore disposal recommendations for all model categories.")

    for category in class_names:

        title, advice = disposal_advice(category)

        with st.expander(f"♻️ {display_name(category)}"):

            st.markdown(f"**{title}**")
            st.write(advice)


# ==========================================
# ABOUT MODEL
# ==========================================

elif page == "About Model":

    st.title("🧠 About the AI Model")

    st.markdown("""
    Smart Waste AI is an educational deep learning project
    that classifies waste images into 12 categories.

    ### Model Architecture

    **MobileNetV2**

    MobileNetV2 is a lightweight convolutional neural network
    architecture designed for efficient image classification.

    ### Training Details

    | Parameter | Value |
    |---|---|
    | Architecture | MobileNetV2 |
    | Image size | 224 × 224 |
    | Training epochs | 5 |
    | Training images | 12,412 |
    | Validation images | 3,103 |
    | Best validation accuracy | 96.07% |
    | Hardware used | NVIDIA GTX 1650 |
    | Framework | PyTorch |

    ### Supported Categories

    - Battery
    - Biological
    - Brown Glass
    - Cardboard
    - Clothes
    - Green Glass
    - Metal
    - Paper
    - Plastic
    - Shoes
    - Trash
    - White Glass

    ### Dataset

    Garbage Classification dataset.

    ### Technology Stack

    - Python
    - PyTorch
    - Torchvision
    - Streamlit
    - Pillow

    ### Project Objective

    To demonstrate how deep learning can support waste
    identification and encourage responsible waste segregation.

    ### Disclaimer

    This is an academic prototype. Model predictions can be
    incorrect. Always follow local municipal disposal rules.
    """)

    st.info(f"Model device: {DEVICE} | Input image size: {image_size} × {image_size}")