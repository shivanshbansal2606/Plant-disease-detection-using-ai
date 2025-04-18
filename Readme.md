
# 🌿 Plant Disease Recognition System

An intelligent deep learning-powered web app to detect plant diseases from leaf images using a Convolutional Neural Network (CNN), built with **TensorFlow** and **Streamlit**.

---

## 🚀 Features

- Upload plant leaf images to detect disease or health status.
- Supports **38 different plant diseases and healthy conditions**.
- Displays the **confidence** of the model's prediction.
- Built with a user-friendly **Streamlit** interface.

---

## 📦 Requirements

Install dependencies:

```bash
pip install streamlit tensorflow numpy pillow matplotlib
```

---

## 🧠 Model Information

- **Model**: Convolutional Neural Network (CNN)
- **Input Size**: 128x128 RGB images
- **Output Classes**: 38
- **Optimizer**: Adam (`lr=0.0001`)
- **Loss Function**: Categorical Crossentropy
- **Training Epochs**: 10
- **Accuracy**: ~97%
- **Trained On**: 70,295 images (with 17,572 validation)

---

## 🧾 Dataset

- [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- **Image Count**: 87,000+
- **Categories**: 38
- **Augmentation**: Offline image augmentation used
- **Preprocessing**: Images resized to 128x128 pixels

---

## 📁 Files

- `trained_plant_disease_model.h5` — Trained Keras model
- `accuracy.png` — Accuracy curve for the model
- `app.py` — Streamlit app script

---

## 🖥️ Running the App

```bash
streamlit run app.py
```

Then, open the local URL provided by Streamlit in your browser.

---

## 📊 Prediction Output

After uploading a leaf image, the app will:

- Display the predicted **disease or health status**
- Show **confidence percentage**
- Preview the **uploaded image**

---

## 💻 Device Info (Used for Training & Testing)

- **Device**: MacBook Air M1 (2020)
- **CPU**: Apple M1 (8-core)
- **GPU**: Integrated 7-core
- **RAM**: 8 GB Unified Memory
- **Dataset Size**: ~2 GB

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

