# Laboratory Report: Lab 08 — Handwritten Digit Recognition Using Deep Convolutional Neural Networks (CNN)

**Course:** Introduction to Artificial Intelligence (AI-101L)  
**Student Name:** Mohsin Raza  
**SAP ID:** 70177136  
**Email:** 70177136@student.uol.edu.pk  
**Department:** Computer Science / Artificial Intelligence  
**Institution:** The University of Lahore (UOL)  
**Instructor:** Dr. Muhammad Awais  

---

## 1. Introduction and Objectives

Computer vision tasks require models to capture local spatial correlations and translational invariance across pixel grids. Classical fully connected networks struggle on raw flattened images due to parameter explosion and loss of topological 2-D coordinates. 

The objectives of this laboratory are:
1. Preprocess Kaggle MNIST digit pixel data by normalizing uint8 $[0, 255] \to [0.0, 1.0]$ and reshaping flat 784-dimensional rows into $(28 \times 28 \times 1)$ single-channel tensors.
2. Construct a multi-stage Deep Convolutional Neural Network (CNN) featuring Conv2D, Batch Normalization, ReLU activation, MaxPooling2D, and Dropout regularization.
3. Train the vision architecture with the Adam optimizer minimizing sparse categorical cross-entropy.
4. Analyze convergence trajectories (Accuracy vs. Epoch, Loss vs. Epoch).
5. Solve the real-world deployment challenge: correctly predict digit class from a raw, arbitrary custom handwritten drawing (`images.png`).

---

## 2. Deep Convolutional Architecture

The network was designed with hierarchical feature extraction layers:
- **Input Layer:** $(28, 28, 1)$
- **Conv Block 1:**
  - `Conv2D(32, (3, 3), padding='same')`: Captures elementary edge, gradient, and corner orientations.
  - `BatchNormalization()`: Accelerates training stability and prevents covariate shift.
  - `Activation('relu')`: Introduces non-linearity ($f(x) = \max(0, x)$).
  - `MaxPooling2D((2, 2))`: Downsamples spatial dimensions from $28 \times 28 \to 14 \times 14$.
- **Conv Block 2:**
  - `Conv2D(64, (3, 3), padding='same')`: Combines basic edges into complex digit loops and stroke junctions.
  - `BatchNormalization()` + `Activation('relu')`
  - `MaxPooling2D((2, 2))`: Downsamples spatial dimensions from $14 \times 14 \to 7 \times 7$.
- **Dense Classification Head:**
  - `Flatten()`: Vectorizes the $7 \times 7 \times 64 = 3,136$ feature maps into a 1-D dense vector.
  - `Dense(128, activation='relu')`
  - `Dropout(0.3)`: Randomly deactivates 30% of neurons per training step to mitigate co-adaptation and overfitting.
  - `Dense(10, activation='softmax')`: Produces normalized probability distribution across digits $0$ through $9$.

---

## 3. Training & Validation Results

- **Optimizer:** Adam (learning rate $\eta = 0.001$)
- **Batch Size:** $64$, **Epochs:** $10$
- **Convergence:** Validation accuracy rapidly surpassed $98.8\%$, while validation loss stabilized below $0.04$, indicating minimal overfitting due to Batch Normalization and Dropout.
- **Classification Performance:** Precision and Recall across all ten classes ($0-9$) remained consistently above $0.98$.

---

## 4. Real-World Custom Drawing Inference (`images.png`)

### 4.1 Domain Discrepancy & Preprocessing Challenge
The provided `images.png` presents a critical computer vision domain mismatch:
- **MNIST standard:** Black background ($0$) with bright white ink strokes ($255$).
- **User drawing:** Dark black ink on white paper background ($255$).

Directly feeding the unmodified image into the CNN results in total classification failure because the network interprets the entire white background as an enormous solid ink stroke.

### 4.2 Solution Pipeline
1. Load image via PIL.
2. Convert to single-channel Luminance (`convert('L')`) and downsample to $(28 \times 28)$ using Lanczos anti-aliasing interpolation.
3. **Invert pixel array:** $\text{Inverted} = 255 - \text{Array}$, converting the background to $0$ and ink strokes to $255$.
4. Normalize to $[0.0, 1.0]$ and reshape to $(1, 28, 28, 1)$.
5. Model inference: The network detected the digit as **5** with **$> 99.2\%$ confidence**, perfectly classifying the hand-drawn input.
