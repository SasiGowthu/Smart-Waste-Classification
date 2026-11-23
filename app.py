from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


MODEL_PATH = Path("models/waste_classifier.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="Smart Waste Classification", page_icon="♻️", layout="centered")


@st.cache_resource
def load_classifier():
    if not MODEL_PATH.exists():
        return None, None, None

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    class_names = checkpoint["class_names"]
    image_size = checkpoint.get("image_size", 224)

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, class_names, preprocess


def disposal_advice(label):
    key = label.lower().replace("_", " ").replace("-", " ")
    if "biological" in key or "organic" in key:
        return "Place in the wet / organic waste stream. Compost suitable food and garden waste where local facilities allow."
    if "paper" in key or "cardboard" in key:
        return "Keep it clean and dry, then place it with paper and cardboard recyclables."
    if "plastic" in key:
        return "Empty and rinse when practical, then place in the plastic recycling stream accepted by your local collection service."
    if "metal" in key or "battery" in key:
        return "Keep separate from ordinary mixed waste. Take batteries to an authorised battery collection point; put accepted metal in dry recyclables."
    if "glass" in key:
        return "Keep glass separate and use your local glass collection point. Handle broken glass carefully and wrap it securely."
    if "clothes" in key or "shoes" in key:
        return "Donate or reuse if still usable. Otherwise check local textile collection or recycling options."
    if "trash" in key:
        return "This category may be residual waste. Check whether the item can be reused or recycled locally before disposal."
    return "Check your local municipal waste-segregation rules for the correct disposal stream."


st.title("♻️ Smart Waste Classification")
st.write("Upload a waste image to predict its category and see a suggested disposal method.")
st.info("This is an educational prototype. Predictions can be wrong; follow your local municipal waste rules.")

model, class_names, preprocess = load_classifier()

if model is None:
    st.error("Trained model not found. First run `python train.py` from the project folder.")
    st.stop()

uploaded_file = st.file_uploader("Choose a waste image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Classify Waste", type="primary"):
        input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probabilities = torch.softmax(model(input_tensor), dim=1)[0]

        confidence, predicted_index = torch.max(probabilities, dim=0)
        predicted_label = class_names[predicted_index.item()]

        st.subheader("Prediction")
        st.success(f"Category: {predicted_label.replace('_', ' ').title()}")
        st.metric("Model confidence", f"{confidence.item() * 100:.2f}%")
        st.subheader("Disposal suggestion")
        st.write(disposal_advice(predicted_label))

        st.subheader("Top predictions")
        top_count = min(3, len(class_names))
        top_probs, top_indices = torch.topk(probabilities, top_count)
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            st.write(f"**{class_names[idx].replace('_', ' ').title()}** — {prob * 100:.2f}%")
            st.progress(float(prob))
