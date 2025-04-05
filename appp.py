import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the model
model = load_model('first_model.h5')

# Get model input shape (e.g., (None, 128, 128, 3))
input_shape = model.input_shape[1:3]  # Extract the expected input size (height, width)

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize(input_shape)  # Resize the image to the model's expected input size
    img_array = np.array(img)        # Convert the image to a numpy array
    img_array = img_array / 255.0    # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to predict using the model
def predict_image(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    return predictions

# Custom function to create a scale with a marker
def show_confidence_scale(label, confidence_percent):
    st.write(f"**{label}: {confidence_percent:.2f}%**")
    
    # HTML and CSS for the confidence scale
    scale_html = f"""
    <div style="width: 100%; background-color: #e0e0e0; height: 10px; position: relative; border-radius: 5px;">
        <div style="width: {confidence_percent}%; background-color: #4CAF50; height: 100%; border-radius: 5px;"></div>
        <div style="position: absolute; top: -5px; left: {confidence_percent}%; transform: translateX(-50%);">
            <span style="position: absolute; top: -20px;">{confidence_percent:.2f}%</span>
            <div style="width: 5px; height: 20px; background-color: #4CAF50;"></div>
        </div>
    </div>
    """
    
    # Render the scale
    st.markdown(scale_html, unsafe_allow_html=True)

# Streamlit App
st.title('Cataract Detection')

# Upload image
uploaded_file = st.file_uploader("Choose an eye image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write("")
    st.write("Classifying...")

    # Predict the result
    try:
        predictions = predict_image(image)
        
        # Debug: Print the prediction output
        st.write(f"Model Prediction Output: {predictions}")
        st.write(f"Prediction Shape: {predictions.shape}")

        # Check if the model returns a single value (binary classification)
        if predictions.shape[1] == 1:
            # Single output: probability for the "Normal" class
            normal_confidence = predictions[0][0]  # Confidence for "Normal"
            cataract_confidence = 1 - normal_confidence  # Complement for "Cataract"
        else:
            # Multi-class output: probabilities for "Normal" and "Cataract"
            normal_confidence = predictions[0][0]  # Confidence for "Normal"
            cataract_confidence = predictions[0][1]  # Confidence for "Cataract"

        # Convert confidence to percentage
        normal_confidence_percent = normal_confidence * 100
        cataract_confidence_percent = cataract_confidence * 100

        # Prediction logic
        if normal_confidence_percent >= 50:
            predicted_class = 'Normal'
        else:
            predicted_class = 'Cataract'

        st.write(f"Prediction: **{predicted_class}**")

        # Show custom confidence scales
        show_confidence_scale('Normal', normal_confidence_percent)
        show_confidence_scale('Cataract', cataract_confidence_percent)

    except ValueError as e:
        st.error(f"Error: {str(e)}. Please check the input size or model compatibility.")
