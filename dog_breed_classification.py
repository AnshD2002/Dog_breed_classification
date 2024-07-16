import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import streamlit as st

# Function to check prediction
def check_prediction(img, model, class_names):
    # Resize the image if not already 224x224
    if img.size != (224, 224):
        img = img.resize((224, 224))

    # Convert the image to array and preprocess
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Make a prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_label = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    return predicted_label, confidence

# Load model
model = tf.keras.models.load_model("Dog_breed_classification_model.keras")
class_names = ['Beagle', 'Boxer', 'Bulldog', 'Dachshund', 'German_Shepherd', 'Golden_Retriever', 'Labrador_Retriever', 'Poodle', 'Rottweiler', 'Yorkshire_Terrier']

# Streamlit app
st.title("Dog Breed Classification")
st.write("Upload an image of a dog, and the model will predict its breed.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Make prediction
    predicted_label, confidence = check_prediction(img, model, class_names)

    # Display prediction
    st.write(f"Prediction: {predicted_label} ({confidence:.2f}%)")

else:
    st.write("There is an error.....")
