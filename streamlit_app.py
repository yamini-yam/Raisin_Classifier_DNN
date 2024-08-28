import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# Load the trained model and scaler
model = tf.keras.models.load_model('Raisin_Classifier.h5')
scaler = joblib.load('scaler.pkl')

# Add some visual appeal with images
st.image("RAISIN_BACKGROUND.webp", caption="Raisin Image", use_column_width=True)

# Streamlit app title and description
st.title("Raisin Classification")
st.markdown("### Predict whether a raisin is of type **Kecimen** or **Besni** based on its features.")

# Sidebar for user input
st.sidebar.header("Input Raisin Features")

def user_input_features():
    area = st.sidebar.slider("Area", 33565, 90856, 57291)
    major_axis = st.sidebar.slider("Major Axis Length", 261.55, 442.27, 352.0)
    minor_axis = st.sidebar.slider("Minor Axis Length", 167.71, 291.36, 229.5)
    eccentricity = st.sidebar.slider("Eccentricity", 0.51, 0.86, 0.68)
    convex_area = st.sidebar.slider("Convex Area", 35794, 93717, 64755)
    extent = st.sidebar.slider("Extent", 0.64, 0.79, 0.72)
    perimeter = st.sidebar.slider("Perimeter", 751.41, 1208.58, 980.0)
    
    data = {
        "Area": area,
        "MajorAxisLength": major_axis,
        "MinorAxisLength": minor_axis,
        "Eccentricity": eccentricity,
        "ConvexArea": convex_area,
        "Extent": extent,
        "Perimeter": perimeter
    }
    return pd.DataFrame(data, index=[0])

# Get user input
user_df = user_input_features()

# Predict button
if st.button("Predict"):
    # Standardize user input
    user_scaled = scaler.transform(user_df)
    
    # Predict the class (0 for Kecimen, 1 for Besni)
    predicted_class = model.predict(user_scaled)[0][0]
    
    # Display the prediction
    if predicted_class < 0.5:
        st.markdown("<h2 style='color: #28a745;'>Predicted class: Kecimen</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='color: #d9534f;'>Predicted class: Besni</h2>", unsafe_allow_html=True)



# Display the user inputs
st.subheader("User Input Features")
st.write(user_df)
