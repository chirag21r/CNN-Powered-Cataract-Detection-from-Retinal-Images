import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from io import BytesIO


CLASS_NAMES = ['Cataracts', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']


def main():
    st.title('EYE DISEASE CLASSIFICATION')

    with st.expander("Instruction"):
        st.markdown("## Description")
        st.markdown("""
        - This is an eye disease classification system that allows users to upload retinal images and perform predictions.
        - The system can predict the following classes: Cataracts, Diabetic Retinopathy, Glaucoma, and Normal.
        """)

        st.markdown("## Procedure")
        st.markdown("""
        1. Click the "Upload Image" button to upload a retinal image.
        2. Click the "Predict Eye Disease" button to start the eye disease prediction.
        3. Please wait for a few moments to get the result.
        4. The result will be displayed below the image.
        """)

        st.markdown("## Accuracy")
        st.markdown("""
        - <p><span style="color: Green;"><strong> Green </strong></span> -- Indicate strong confidence</p>
        - <p><span style="color: Red;"><strong> Red </strong></span> -- Indicate weak confidence</p>
        """, unsafe_allow_html=True)


    model = load_model()
    image = load_image()
    result = st.button('PREDICT EYE DISEASE')



    if result:

        result_placeholder = st.empty()
        result_placeholder.write('Calculating results...')

        predicted_class, accuracy_percentage = predict(model, CLASS_NAMES, image)

        result_placeholder.empty() 

        if predicted_class and accuracy_percentage:
            st.markdown(f'<p><strong>Predicted Class:</strong> {predicted_class}</p>', unsafe_allow_html=True)
            if accuracy_percentage >= 0.80:
                st.markdown(f'<p><strong>Accuracy:</strong> <span style="color: Green;"><strong>{accuracy_percentage}%</strong></span></p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p><strong>Accuracy:</strong> <span style="color: Red;"><strong>{accuracy_percentage}%</strong></span></p>', unsafe_allow_html=True)





def load_model():
    model = tf.keras.models.load_model('model/VGG19/model.epoch06-loss0.34.h5')
    return model


def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)

        temp = Image.open(BytesIO(image_data))
        print('dwd', temp)

        return Image.open(BytesIO(image_data))
    else:
        return None


def predict(model, class_names, image):
     # Resizing the image
    resized_image = image.resize((224, 224))

    # Convert PIL image into array
    image = np.array(resized_image)

    # # Normalize the image
    normalized_image = image / 255.0

    input_image = np.expand_dims(normalized_image, axis=0)

    predictions = model.predict(input_image)
    print(predictions)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    accuracy = np.max(predictions[0])
    accuracy_percentage = '{:.2}'.format(accuracy)


    return predicted_class, float(accuracy_percentage)








if __name__ == '__main__':
    main()
