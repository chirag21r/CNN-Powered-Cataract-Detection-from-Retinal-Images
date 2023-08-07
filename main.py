import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import KMeans
from io import BytesIO


CLASS_NAMES = ['Cataracts', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']


def main():
    st.title('EYE DISEASE CLASSIFICATION')

    with st.expander("Instruction"):
        st.markdown("## Description")
        st.markdown("""
        - This is an eye disease classification system that allows users to upload retinal images and perform predictions.
        - The system can predict the following classes: <p>
                    <span style="color: Red;"><strong>Cataracts</strong></span>, 
                    <span style="color: Red;"><strong>Diabetic Retinopathy</strong></span>, 
                    <span style="color: Red;"><strong>Glaucoma</strong></span>, 
                    <span style="color: Red;"><strong>Normal</strong></span>.</p>
        """, unsafe_allow_html=True)

        st.markdown("## Steps")
        st.markdown("""
        1. Click the "Browse files" button to upload a retinal image.
        2. Click the "Predict Eye Disease" button to start the eye disease prediction.
        3. Please wait for a few moments to get the result.
        4. The result will be displayed below the image.
        </br>
        """, unsafe_allow_html=True)

        st.markdown("## Confidence")
        st.markdown("""
        - Confidence is determined by the probabilities obtained from the softmax layer of the deep learning model.
        - The softmax layer converts the final output of the model into probability scores, reflecting the model's confidence in its predictions for each class in the multiclass classification.
        """)
        
        


    model = load_model()
    image = load_image()
    result = st.button('PREDICT EYE DISEASE')



    if result:

        result_placeholder = st.empty()
        result_placeholder.write('Calculating results...')

        preprocessed_image = preprocess_image(image)
        predicted_class, rounded_percentage = predict(model, CLASS_NAMES, preprocessed_image)

        result_placeholder.empty() 

        if predicted_class and rounded_percentage:
            st.markdown(f'<p><strong>Predicted Class:</strong> {predicted_class}</p>', unsafe_allow_html=True)
            st.markdown(f'<p><strong>Confidence:</strong> <span style="color: Blue;"><strong>{rounded_percentage}%</strong></span></p>', unsafe_allow_html=True)
            
           





def load_model():
    model = tf.keras.models.load_model('model/VGG19/model.epoch06-loss0.34.h5')
    return model


def load_image():
    uploaded_file = st.file_uploader(label='Pick an retinal image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        
        # Display image into streamlit
        st.image(image_data)

        # Load image with BytesIO
        temp = Image.open(BytesIO(image_data))

        # Resizing the image
        resized_image = temp.resize((224, 224))

        # Convert PIL image into array
        image = np.array(resized_image)
        
        return image
    else:
        return None


def preprocess_image(image):
    # Histogram Equalization Part

    # Split into 3 channels
    red, green, blue = cv2.split(image)

    # Apply Histogram Equalization in RED
    equalization = cv2.equalizeHist(red)

    # Merge back the channels
    merged_image = cv2.merge((equalization, green, blue))


    # Image Segmentation Part
    segmentation = merged_image.reshape(-1, 3)

    # Declare K-means clustering image segmentation
    kmeans = KMeans(n_clusters=20, n_init=5)

    # Perform segmentation for the images
    kmeans.fit(segmentation)

    segmented_images = kmeans.cluster_centers_[kmeans.labels_]
    segmented_images = segmented_images.reshape(merged_image.shape)
    # st.image(segmented_images.astype("uint8"))

    return segmented_images



def predict(model, class_names, image):
    # Normalize the image
    normalized_image = image / 255.0

    input_image = np.expand_dims(normalized_image, axis=0)

    predictions = model.predict(input_image)
    print(predictions)

    predicted_class = class_names[np.argmax(predictions)]
    accuracy = np.max(predictions)
    accuracy_percentage = accuracy * 100
    rounded_percentage = round(accuracy_percentage, 2)
    print(rounded_percentage)


    return predicted_class, rounded_percentage






if __name__ == '__main__':
    main()
