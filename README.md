This will include ML models 
#  Machine Learning Model Zoo

Welcome to my **Machine Learning Model Zoo** — a centralized repository where I explore and test a variety of machine learning and deep learning models. This collection showcases hands-on experiments with different datasets and architectures, intended for learning, experimentation, and showcasing to recruiters and collaborators.

---

##  Projects Included
file name:= RNN_mnist.ipynb
1) Handwritten Digit Recognition (MNIST - RNN)

- **Dataset**: [MNIST](http://yann.lecun.com/exdb/mnist/)
- **Model**: Simple Recurrent Neural Network (RNN)
- **Framework**: TensorFlow / Keras
- **Objective**: Classify 28×28 grayscale images of handwritten digits (0–9).
- **Highlights**:
  - Preprocessing and reshaping image data for RNNs
  - One-hot encoding of labels
  - Visualization of training accuracy and loss
  - Random sample prediction with evaluation



---

### 2. Breast Cancer Detection (Binary Classification)
file name:= Breast_cancer_NN.ipynb
- **Dataset**: Scikit-learn’s Breast Cancer Wisconsin dataset
- **Model**: Feedforward Neural Network
- **Framework**: TensorFlow / Keras
- **Objective**: Predict if a tumor is **malignant** or **benign** based on 30 features.
- **Highlights**:
  - Feature scaling using `StandardScaler`
  - Neural network architecture with multiple Dense layers
  - Binary classification using sigmoid activation



---

3) Obstacle avoiding game
   file := Obstacle_game_RL.ipynb
uses reinforcement learning techniques and concepts to achieve the game
uses matplotlib for GUI


---
MNIST Noise Injection (for Denoising Autoencoders)
Filename:= encoderb.ipynb
Dataset: MNIST

Objective: Simulate noisy digit images to prepare data for training a denoising autoencoder.

Highlights:

Adds Gaussian noise to selected MNIST images

Visualizes original vs noisy images

Useful for training models that learn to recover clean images from corrupted ones



## 🚀 Future Additions

This repo will grow to include models like:
- CNNs for image classification
- LSTM/GRU for sequence prediction
- Regression models
- Clustering (e.g., KMeans)
- NLP tasks (Sentiment analysis, text classification)

---

## 🛠️ Requirements

- Python 3.x
- TensorFlow / Keras
- scikit-learn
- numpy / pandas
- matplotlib

Install dependencies:
```bash
pip install -r requirements.txt
