import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Page config
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

# Title
st.title("🧠 Brain Tumor Detection System")
st.write("Upload an MRI image to check for brain tumor presence.")

# Load model
@st.cache_resource
def load_trained_model():
    model = load_model("braintumer.h5")
    return model

model = load_trained_model()

IMG_SIZE = 224

# Image preprocessing
def preprocess_image(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


# File uploader
uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing MRI Image..."):
            img_array = preprocess_image(img)
            prediction = model.predict(img_array)[0][0]

        st.subheader("Prediction Result")

        if prediction > 0.5:
            st.error("🧠 Tumor Detected")
            st.write(f"Confidence Score: **{prediction:.2f}**")
        else:
            st.success("✅ No Tumor Detected")
            st.write(f"Confidence Score: **{1 - prediction:.2f}**")

