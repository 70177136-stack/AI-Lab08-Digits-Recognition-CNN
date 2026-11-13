# Lab 08: Handwritten Digit Recognition with Deep CNNs

**Author:** Mohsin Raza  
**SAP ID:** 70177136  
**Email:** 70177136@student.uol.edu.pk  
**Institution:** The University of Lahore (UOL)  
**Course:** Introduction to Artificial Intelligence (AI-101L) / Deep Learning  

---

## 📌 Overview
This repository contains the complete deep learning pipeline for **Laboratory 08**, training a 2D Convolutional Neural Network (CNN) in TensorFlow/Keras on the MNIST Digit Recognizer dataset and testing on real-world custom hand-drawn digit sketches.

---

## 📁 Repository Structure
```
Lab08_Digits_Recognition_CNN/
├── Digits_Recognition.ipynb     # Complete Jupyter Notebook with training and inference
├── images.png                   # Custom hand-drawn digit test image (Digit '5')
├── train.csv                    # MNIST training dataset (Kaggle)
├── test.csv                     # MNIST test dataset (Kaggle)
├── report.md                    # Detailed laboratory report & architecture analysis
├── requirements.txt             # Python dependencies
└── README.md                    # Execution guide
```

---

## 🚀 Execution Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Jupyter Notebook
```bash
jupyter notebook Digits_Recognition.ipynb
```
The notebook executes sequentially from data loading, normalization, CNN training, learning curve generation, Kaggle test set predictions, through custom inference on `images.png`.
