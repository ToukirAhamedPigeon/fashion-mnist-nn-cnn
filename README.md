# Fashion MNIST: Neural Network vs Convolutional Neural Network

This project compares the performance of a **simple Neural Network (NN)** and a **Convolutional Neural Network (CNN)** on the **Fashion MNIST dataset**. It includes full data preprocessing, model training, evaluation, visualizations, and analysis.

---

## Project Overview

- **Dataset:** Fashion MNIST (60,000 train, 10,000 test images, 28×28 grayscale)
- **Classes:** 10 clothing categories
- **Objective:** Compare NN vs CNN performance, analyze accuracy, and visualize predictions

---

## Folder Structure
fashion-mnist-nn-cnn/
├── models.py # Contains NN and CNN model code
├── requirements.txt # Python dependencies
├── .gitignore # Ignore venv, pycache, etc.
├── README.md # Project documentation
└── venv/ # Optional local virtual environment


---

## Hardware Requirements

- **CPU:** Minimum 4 cores, 8GB RAM  
- **GPU (optional but recommended):** NVIDIA GPU with CUDA support for faster training  
- **Python:** ≥3.8  
- **Libraries:** TensorFlow, NumPy, Matplotlib, scikit-learn

---

## Environment Setup

1. **Clone repository:**
git clone <your-repo-url>
cd fashion-mnist-nn-vs-cnn
2. **Create and activate virtual environment:**
python -m venv venv
# Activate venv (Windows)
venv\Scripts\activate
# Activate venv (Linux/Mac)
source venv/bin/activate
3. **Install dependencies:**
pip install -r requirements.txt

## Running the Project
1. Open models.py in an IDE or terminal.
2. Run the script:
    python models.py
3. The script will:
    - Load and preprocess Fashion MNIST
    - Train NN and CNN models
    - Evaluate test accuracy
    - Plot sample predictions and training curves
## Model Architecture
1. Neural Network (NN)
    - Input: 28×28 flattened → 784 neurons
    - Dense Layer 1: 128 neurons, ReLU
    - Dense Layer 2: 64 neurons, ReLU
    - Output: 10 neurons, softmax
2. Convolutional Neural Network (CNN)
    - Conv2D 32 filters, 3×3 kernel, ReLU
    - MaxPooling2D 2×2
    - Conv2D 64 filters, 3×3 kernel, ReLU
    - MaxPooling2D 2×2
    - Flatten → Dense 128 neurons, ReLU → Dropout 0.4
    - Output: 10 neurons, softmax
## Project Results
- Test accuracy of NN and CNN
- Sample predictions with images
- Training and validation curves
- Confusion matrices
- Observation: CNN typically outperforms NN due to spatial feature extraction.

## Google Colab
You can also run the project in Colab:
https://colab.research.google.com/drive/1hrYbtYi0iu_pgEoxlXonW8nJB_uRPx7g?usp=sharing
