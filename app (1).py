
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Define the path to your saved model
# Make sure this path is correct and accessible
MODEL_PATH = '/content/drive/MyDrive/my_cnn_model.keras'

# Define the image dimensions your model expects
IMG_HEIGHT = 128
IMG_WIDTH = 128

# Define the class names in the same order as your model was trained
# Replace with your actual class names
CLASS_NAMES = ['acai', 'cupuacu', 'graviola', 'guarana', 'pupunha', 'tucuma']

# Load the trained model
@st.cache_resource # Cache the model to avoid reloading on each interaction
def get_model():
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = get_model()

st.title("Image Classification App")

if model is not None:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        img = image.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
        img_array = img_array / 255.0 # Normalize to [0, 1]

        # Make a prediction
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = np.max(predictions)

        st.write(f"Prediction: {predicted_class_name}")
        st.write(f"Confidence: {confidence:.2f}")

else:
    st.write("Model could not be loaded. Please check the model path.")
