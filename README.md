🧬 Skin Cancer Detection Dashboard

An AI-based Skin Cancer Detection System built using a Convolutional Neural Network (CNN) and an interactive Streamlit dashboard.
The application allows users to upload dermoscopic skin lesion images and receive a prediction (Benign or Malignant) along with a confidence score and training insights.

📌 Project Overview

Skin cancer is one of the most common types of cancer worldwide. Early detection can significantly improve survival rates.
This project demonstrates how deep learning and computer vision can assist in the early screening of skin cancer using medical images.

The system:

Uses a CNN trained on dermoscopic images

Provides real-time predictions through a web dashboard

Displays model confidence and training performance

🚀 Features

📤 Upload skin lesion images (JPG, JPEG, PNG)

🧠 CNN-based classification:

Benign (Non-cancerous)

Malignant (Cancerous)

📊 Confidence score with progress bar

📈 Training accuracy & loss visualization

🎨 Modern glassmorphism UI with gradient theme

📄 Downloadable PDF report (educational use)

📂 Clean project structure suitable for portfolio

🛠️ Tech Stack

Programming Language: Python

Deep Learning: TensorFlow, Keras

Image Processing: PIL, NumPy

Web Framework: Streamlit

Visualization: Matplotlib

PDF Generation: FPDF

📂 Project Structure
skin-cancer-detection-dashboard/
│
├── dashboard.py               # Streamlit application
├── train_skin_cancer.py       # CNN model training script
├── images/                    # Dataset (train / validation)
├── training_curves.png        # Accuracy & loss curves
├── Figure_1.png               # Sample output image
├── README.md                  # Project documentation
├── .gitignore                 # Ignored files & folders
└── streamlit                  # Streamlit config (if any)

🧠 Model Details

Architecture: Convolutional Neural Network (CNN)

Input Size: 224 × 224 RGB images

Output: Binary classification (Benign / Malignant)

Loss Function: Binary Crossentropy

Optimizer: Adam

Evaluation Metrics: Accuracy, Validation Loss

▶️ How to Run the Project Locally

1️⃣ Clone the Repository

git clone https://github.com/Thrisha170/skin-cancer-detection-dashboard.git

cd skin-cancer-detection-dashboard

2️⃣ (Optional) Create a Virtual Environment

python -m venv venv

venv\Scripts\activate   # Windows

3️⃣ Install Required Packages

pip install tensorflow streamlit pillow numpy matplotlib fpdf

4️⃣ Run the Streamlit App

streamlit run dashboard.py


The app will open in your browser at:

http://localhost:8501

📊 Training Insights

The dashboard includes:

Training Accuracy vs Validation Accuracy

Training Loss vs Validation Loss

These help understand:

Model learning behavior

Overfitting / underfitting trends

📄 PDF Report

The application generates a downloadable PDF report containing:

Prediction result

Confidence score

Disclaimer for educational use

(Note: This project is intended for learning and demonstration purposes.)

⚠️ Disclaimer

⚠️ This application is for educational and demonstration purposes only.
It is not a medical diagnostic tool and should not be used for clinical decisions.

👩‍💻 Author

Kamatchiammal T
GitHub: https://github.com/Thrisha170

🌟 Future Improvements

Deploy on Streamlit Cloud

Add Grad-CAM for explainable AI

Improve PDF report with images & metadata

Add multi-class skin lesion classification

Improve model accuracy with data augmentation

⭐ Portfolio Note

This project demonstrates:

Practical use of Deep Learning

End-to-end ML pipeline

Real-time AI dashboard development

Git & GitHub workflow

Debugging and deployment readiness
