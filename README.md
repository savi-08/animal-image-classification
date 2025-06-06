# 🐾 Animal Image Classification using CNN & Transfer Learning

This project uses **Transfer Learning (MobileNetV2)** and **Convolutional Neural Networks (CNN)** to classify animal images into 15 different categories such as cat, dog, lion, panda, and more.

---
## 📁 Dataset Structure

The dataset is organized as follows:

dataset/
└── train/
    ├── Cat/
    ├── Dog/
    ├── Elephant/
    ├── Lion/
    ├── Panda/
    └── ... (15 classes total)
---

## 🚀 Features

- ✅ MobileNetV2 (Transfer Learning)
- ✅ Data Augmentation
- ✅ Dropout layers to reduce overfitting
- ✅ Model saved as `.h5`
- ✅ CLI-based image prediction (`predict.py`)

---

## 🛠️ How to Run

```bash
# Clone the repo
git clone https://github.com/savi-08/animal-image-classification.git
cd animal-image-classification

# Setup environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Train the model
python train_transfer.py

# Make predictions
python predict.py

Enter the image filename (e.g., test_dolphin.jpg):
✅ Prediction: Dolphin

Made by Shravani Bande
📍 Akola, Maharashtra
🔗 https://github.com/savi-08