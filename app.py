import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

# Directory to save uploaded images with predicted labels
SAVE_DIR = "saved_predictions"  # Define the directory to save images
os.makedirs(SAVE_DIR, exist_ok=True)  # Create the directory if it doesn't exist

# TensorFlow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Function to save uploaded image with its predicted label
def save_uploaded_image(image, disease_name):
    # Convert the image to a PIL image
    pil_image = Image.open(image)
    
    # Generate a unique filename using the predicted disease name and timestamp
    filename = f"{disease_name}_{image.name}"
    
    # Save the image with the predicted disease label in the specified directory
    save_path = os.path.join(SAVE_DIR, filename)
    pil_image.save(save_path)
    st.write(f"Image saved as: {save_path}")

# Function to provide simple solutions based on predicted disease
def get_disease_solution(disease_name):
    solutions = {
        'Apple___Apple_scab': 'Spray fungicides like captan or myclobutanil and prune affected leaves to improve air circulation.',
        'Apple___Black_rot': 'Remove infected fruits and branches. Use fungicides and keep the area around trees clean.',
        'Apple___Cedar_apple_rust': 'Remove nearby juniper plants and use fungicides to protect the apple trees.',
        'Apple___healthy': 'No action needed. Keep monitoring regularly.',
        'Blueberry___healthy': 'Keep the soil acidic, water regularly, and mulch to retain moisture.',
        'Cherry_(including_sour)___Powdery_mildew': 'Use sulfur-based sprays and prune to improve airflow around the plants.',
        'Cherry_(including_sour)___healthy': 'Maintain good watering and fertilization practices.',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Use resistant varieties, crop rotation, and apply fungicides if necessary.',
        'Corn_(maize)___Common_rust_': 'Plant resistant varieties and use fungicides if rust appears on leaves.',
        'Corn_(maize)___Northern_Leaf_Blight': 'Rotate crops and use fungicides to control the disease.',
        'Corn_(maize)___healthy': 'No disease detected. Keep practicing good farming techniques.',
        'Grape___Black_rot': 'Prune infected parts, remove mummified berries, and apply fungicides like mancozeb.',
        'Grape___Esca_(Black_Measles)': 'Prune infected vines and maintain soil health to reduce spread.',
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Apply copper-based fungicides and ensure good air circulation by proper pruning.',
        'Grape___healthy': 'No issues detected. Continue monitoring your grapevines.',
        'Orange___Haunglongbing_(Citrus_greening)': 'Remove infected trees and control the psyllid insects that spread the disease.',
        'Peach___Bacterial_spot': 'Apply copper-based sprays, avoid overhead watering, and remove infected parts.',
        'Peach___healthy': 'Trees are healthy. Continue regular monitoring.',
        'Pepper,_bell___Bacterial_spot': 'Use copper sprays, avoid wetting leaves when watering, and remove infected leaves.',
        'Pepper,_bell___healthy': 'Plants are healthy. Keep them well-watered and monitor regularly.',
        'Potato___Early_blight': 'Use fungicides like chlorothalonil, rotate crops, and remove infected leaves.',
        'Potato___Late_blight': 'Use fungicides, avoid overhead irrigation, and destroy infected plants immediately.',
        'Potato___healthy': 'Potatoes are healthy. Maintain regular care and monitoring.',
        'Raspberry___healthy': 'Plants are healthy. Ensure good watering and nutrient management.',
        'Soybean___healthy': 'No issues detected. Keep monitoring for any changes.',
        'Squash___Powdery_mildew': 'Apply sulfur or potassium bicarbonate sprays and water at the base of the plant to avoid wetting leaves.',
        'Strawberry___Leaf_scorch': 'Remove infected leaves, water early in the day, and avoid overhead watering.',
        'Strawberry___healthy': 'Plants are healthy. Continue to care as usual.',
        'Tomato___Bacterial_spot': 'Use copper-based bactericides and avoid working with wet plants to prevent spreading.',
        'Tomato___Early_blight': 'Remove affected leaves, avoid wetting leaves, and apply fungicides if needed.',
        'Tomato___Late_blight': 'Apply fungicides and remove and destroy infected plants immediately.',
        'Tomato___Leaf_Mold': 'Prune to improve air circulation and apply fungicides like chlorothalonil if necessary.',
        'Tomato___Septoria_leaf_spot': 'Remove infected leaves and apply fungicides regularly to prevent spread.',
        'Tomato___Spider_mites Two-spotted_spider_mite': 'Spray with water to remove mites, or use insecticidal soap if infestation is severe.',
        'Tomato___Target_Spot': 'Use copper-based fungicides and avoid overhead watering.',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Remove infected plants and control whitefly populations that spread the virus.',
        'Tomato___Tomato_mosaic_virus': 'Remove and destroy infected plants. Avoid handling healthy plants after touching infected ones.',
        'Tomato___healthy': 'Tomatoes are healthy. Keep monitoring and maintaining good practices.',
    }
    return solutions.get(disease_name, "No specific solution found for this disease. Consult an agricultural expert for further guidance.")

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System!
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.
                This dataset consists of about 53K RGB images of healthy and diseased crop leaves categorized into 38 different classes. The total dataset is divided into an 80/20 ratio of training and validation sets preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purposes.
                #### Content
                1. train (43,444 images)
                2. test (35 images)
                3. validation (10,862 images)
                """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    
    if test_image is not None:
        st.image(test_image, use_column_width=True)
        
        # Predict button
        if st.button("Predict"):
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            
            # Reading Labels
            class_name = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
            ]
            
            predicted_disease = class_name[result_index]
            st.success(f"Model is Predicting it's a {predicted_disease}")
            
            # Save the uploaded image with the predicted disease name
            save_uploaded_image(test_image, predicted_disease)
            
            # Display the solution for the predicted disease
            with st.expander("View Solution"):
                solution = get_disease_solution(predicted_disease)
                st.info(f"Solution: {solution}")
