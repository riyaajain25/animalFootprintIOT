import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------
# Load your trained model
# ---------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("animal_footprint_best.keras")  # change filename if needed
    return model

model = load_model()

# Define class names (based on your dataset folders)
CLASS_NAMES = ['canid', 'cervid', 'felid', 'ursid']

# ---------------------------
# Preprocessing helper
# ---------------------------
def preprocess_image(image):
    image = image.resize((224, 224))  # same size used in training
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalization
    return img_array

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Animal Footprint Detection", page_icon="🐾", layout="centered")
st.title("🐾 Animal Footprint Detection")
st.write("Upload an image or use live camera to identify the animal footprint.")

# Input source choice
option = st.radio("Choose input source:", ("Upload Image", "Use Mobile Camera"))

# ---------------------------
# Option 1: Upload Image
# ---------------------------
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize((224, 224))
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        img_array = preprocess_image(image)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
        predicted_class = CLASS_NAMES[np.argmax(score)]
        confidence = 100 * np.max(score)
        
        st.success(f"Prediction: **{predicted_class.capitalize()}** 🐾")
        st.write(f"Confidence: **{confidence:.2f}%**")

# ---------------------------
# Option 2: Use Mobile Camera
# ---------------------------
elif option == "Use Mobile Camera":
    st.write("Take a picture using your camera 📸")
    camera_photo = st.camera_input("Click below to capture")

    if camera_photo is not None:
        image = Image.open(camera_photo)
        st.image(image, caption="Captured Image", use_column_width=True)
        
        img_array = preprocess_image(image)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
        predicted_class = CLASS_NAMES[np.argmax(score)]
        confidence = 100 * np.max(score)
        
        st.success(f"Prediction: **{predicted_class.capitalize()}** 🐾")
        st.write(f"Confidence: **{confidence:.2f}%**")
