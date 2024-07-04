import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

# Define custom DepthwiseConv2D class to handle the groups parameter
class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)  # Remove the groups parameter if it exists
        super().__init__(*args, **kwargs)

# Define custom SeparableConv2D class to handle the groups and kernel_initializer parameters
class CustomSeparableConv2D(tf.keras.layers.SeparableConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)  # Remove the groups parameter if it exists
        kwargs.pop('kernel_initializer', None)  # Remove kernel_initializer if it exists
        kwargs.pop('kernel_regularizer', None)  # Remove kernel_regularizer if it exists
        kwargs.pop('kernel_constraint', None)  # Remove kernel_constraint if it exists
        super().__init__(*args, **kwargs)

# Load the pre-trained EEGNet model
@st.cache_resource
def load_eegnet_model():
    try:
        # Custom objects dictionary to pass to load_model
        custom_objects = {
            'DepthwiseConv2D': CustomDepthwiseConv2D,
            'SeparableConv2D': CustomSeparableConv2D
        }
        model = load_model('eegnet_model.h5', custom_objects=custom_objects)
        return model
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

# Function to preprocess user input data
def preprocess_data(delta_power, theta_power, alpha_power, beta_power, gamma_power):
    # Combine the features into a single array
    data = np.array([delta_power, theta_power, alpha_power, beta_power, gamma_power])
    # Create an array with the expected shape (64, 128) and place the data in it
    reshaped_data = np.zeros((64, 128))  # Adjust as needed
    reshaped_data[0, :5] = data  # Place the input data in the correct position
    return reshaped_data.reshape(1, 64, 128, 1)  # Reshape for EEGNet model input shape

# Load the pre-trained model
model = load_eegnet_model()

# Streamlit UI
st.markdown(
    """
    <style>
    .title {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 30px;
    }
    .input-container {
        margin-bottom: 20px;
    }
    .predict-button {
        display: block;
        margin: 20px auto;
        padding: 10px 20px;
        font-size: 18px;
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    .result {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown("<h1 class='title'>EEG Mental State Classification</h1>", unsafe_allow_html=True)
st.markdown("<div class='input-container'>Enter EEG Power Values</div>", unsafe_allow_html=True)

delta_power = st.slider("Delta Power (1-4 Hz)", 0.0, 100.0, 50.0)
theta_power = st.slider("Theta Power (4-8 Hz)", 0.0, 100.0, 50.0)
alpha_power = st.slider("Alpha Power (8-12 Hz)", 0.0, 100.0, 50.0)
beta_power = st.slider("Beta Power (12-30 Hz)", 0.0, 100.0, 50.0)
gamma_power = st.slider("Gamma Power (30-100 Hz)", 0.0, 100.0, 50.0)

if st.button('Predict', key='predict_button'):
    if model is not None:
        input_data = preprocess_data(delta_power, theta_power, alpha_power, beta_power, gamma_power)
        prediction = model.predict(input_data)
        predicted_class = np.argmax(prediction, axis=1)[0]

        class_names = ['Rest', 'Task']
        result = class_names[predicted_class]

        st.markdown(f"<p class='result'>Predicted Mental State: {result}</p>", unsafe_allow_html=True)
    else:
        st.error("Model could not be loaded. Please ensure the 'eegnet_model.h5' file is in the correct directory.")
