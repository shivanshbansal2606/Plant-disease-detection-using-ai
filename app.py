import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json

st.set_page_config(page_title="Plant Disease Recognition", layout="wide")

# Class names for disease prediction
class_name = [
    "Apple - Apple Scab",
    "Apple - Black Rot",
    "Apple - Cedar Apple Rust",
    "Apple - Healthy",
    "Blueberry - Healthy",
    "Cherry (including sour) - Powdery Mildew",
    "Cherry (including sour) - Healthy",
    "Corn (maize) - Cercospora Leaf Spot / Gray Leaf Spot",
    "Corn (maize) - Common Rust",
    "Corn (maize) - Northern Leaf Blight",
    "Corn (maize) - Healthy",
    "Grape - Black Rot",
    "Grape - Esca (Black Measles)",
    "Grape - Leaf Blight (Isariopsis Leaf Spot)",
    "Grape - Healthy",
    "Orange - Haunglongbing (Citrus Greening)",
    "Peach - Bacterial Spot",
    "Peach - Healthy",
    "Pepper, bell - Bacterial Spot",
    "Pepper, bell - Healthy",
    "Potato - Early Blight",
    "Potato - Late Blight",
    "Potato - Healthy",
    "Raspberry - Healthy",
    "Soybean - Healthy",
    "Squash - Powdery Mildew",
    "Strawberry - Leaf Scorch",
    "Strawberry - Healthy",
    "Tomato - Bacterial Spot",
    "Tomato - Early Blight",
    "Tomato - Late Blight",
    "Tomato - Leaf Mold",
    "Tomato - Septoria Leaf Spot",
    "Tomato - Spider Mites / Two-Spotted Spider Mite",
    "Tomato - Target Spot",
    "Tomato - Tomato Yellow Leaf Curl Virus",
    "Tomato - Tomato Mosaic Virus",
    "Tomato - Healthy"
]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_plant_disease_model.h5")

model = load_model()

def model_prediction(test_image):
    image = Image.open(test_image).convert('RGB')
    image = image.resize((128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)
    predictions = model.predict(input_arr)
    return np.argmax(predictions), np.max(predictions)

# App title
st.title("🌿 Plant Disease Recognition System")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📄 Project Information", "🧠 Model Information", "🔍 Prediction", "💻 Device Information"])

# Tab 1: Project Information
with tab1:
    st.markdown("## 🌱 Project Information")

    st.markdown("### 📌 Overview")
    st.write("""
        The **Plant Disease Recognition System** is a deep learning-based solution developed to assist farmers, researchers, and agriculturalists in detecting plant diseases early using just leaf images. By uploading a photo of a plant leaf, the system identifies whether it's healthy or affected by a specific disease, enabling timely action to prevent crop loss.
    """)

    st.markdown("### 🌾 Dataset Details")
    st.write("""
        The model is trained on the **[New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)** from Kaggle, which includes:
        - **Total Images**: 87,000+
        - **Image Type**: RGB (Color)
        - **Categories**: 38 plant disease and healthy leaf classes
        - **Data Split**:
            - **Training Set**: 70,295 images
            - **Validation Set**: 17,572 images
            - **Test Set**: 33 images (used for final predictions)
        - **Image Size**: All images are resized to **128x128 pixels** before feeding into the model.
        - The dataset was created using **offline augmentation** techniques to increase diversity and robustness.

        The original dataset was curated from a public GitHub repository and then restructured for ease of training and evaluation.
    """)

    st.markdown("### 🎯 Project Goal")
    st.write("""
        The main objective of this project is to provide a fast, accurate, and user-friendly plant disease detection tool that:
        - Helps in early identification of diseases
        - Supports 38 different plant disease classes
        - Is accessible via a simple web interface using Streamlit
        - Can potentially assist farmers and plant pathologists in real-time scenarios
    """)

    st.markdown("### ⚙️ Key Technologies Used")
    st.write("""
        - **TensorFlow & Keras**: For building and training the Convolutional Neural Network (CNN)
        - **Streamlit**: For building the interactive web interface
        - **Matplotlib & Seaborn**: For data visualization
        - **NumPy & PIL**: For image processing
        - **Sklearn**: For evaluation metrics like confusion matrix and classification report
    """)

    st.markdown("### 🧪 Testing & Inference")
    st.write("""
        Users can upload leaf images through the Prediction tab to test the model in real-time.
        The model processes the image, predicts the disease category, and shows confidence levels for its prediction.
    """)

# Tab 2: Model Information
with tab2:
    st.markdown("### 🧠 Model Information")

    st.write("**Model Type**: Convolutional Neural Network (CNN)")
    st.write("**Input Shape**: (128, 128, 3)")
    st.write("**Output Classes**: 38 plant disease categories")
    st.write("**Optimizer**: Adam (learning rate = 0.0001)")
    st.write("**Loss Function**: Categorical Crossentropy")
    st.write("**Batch Size**: 32")
    st.write("**Epochs Trained**: 10")
    st.write("**Training Time**: Approximately 78 minutes")

    st.write("**Total Training Images**: 70,295")
    st.write("**Total Validation Images**: 17,572")
    st.write("**Dataset Size**: ~2 GB")

    st.markdown("### 🧩 Model Architecture")
    st.write("""
        The model is based on a Convolutional Neural Network (CNN) architecture. The key layers include:
        - **Convolutional Layers**: To extract features from the images.
        - **Max-Pooling Layers**: To downsample the feature maps and reduce dimensionality.
        - **Fully Connected Layers**: To classify the images based on the extracted features.
        - **Dropout**: To prevent overfitting during training.
    """)

    st.markdown("#### 🔍 Architecture Overview")
    st.markdown("""
    - 5 Convolutional blocks with increasing filters (32 → 512)
    - MaxPooling after each pair of Conv layers
    - Dropout (0.25 after conv blocks, 0.4 after dense)
    - Flatten layer followed by a Dense layer (1500 units)
    - Final Dense output layer with softmax activation
    """)

    st.markdown("### 📈 Model Performance")
    st.write("""
        The model has been trained for 10 epochs and achieved an accuracy of around 97% on the training data.
    """)

    st.image("accuracy.png", caption="Model Accuracy", width=600)

    st.markdown("#### 💾 Model Saved As:")
    st.code("trained_plant_disease_model.keras\ntrained_plant_disease_model.h5")

# Tab 3: Prediction
with tab3:
    st.markdown("Upload a clear image of a plant leaf, and our AI model will analyze it to detect any signs of disease.")

    uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", width=300)

        if st.button("Predict"):
            with st.spinner("Analyzing..."):
                index, confidence = model_prediction(uploaded_file)
                predicted_label = class_name[index]
                st.success(f"🧠 Prediction: **{predicted_label}**")
                st.info(f"📈 Confidence: **{confidence * 100:.2f}%**")

# Tab 4: System Information
with tab4:
    st.markdown("### 💻 Device Information")

    st.write("**Device**: MacBook Air M1 (2020)")
    st.write("**CPU**: Apple M1 (8-core: 4 performance + 4 efficiency cores)")
    st.write("**GPU**: 7-core integrated GPU")
    st.write("**Memory**: 8 GB Unified Memory")
    st.write("**Storage**: 256 GB SSD (≈2 GB used for dataset)")