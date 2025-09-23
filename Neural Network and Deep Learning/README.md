# 🧠 Mental Workload Classification using Deep Learning & Logistic Regression

This project focuses on classifying **mental workload levels** using both **deep learning (CNN)** and a **classical Logistic Regression model**.  
The aim is to compare traditional machine learning techniques with modern neural network architectures for **EEG/mental workload classification**.

---

## 📘 Project Overview
- Built two different models for mental workload classification:
  1. **CNN Model** (`CNNModelForMentalWorkloadClassification.ipynb`)  
     - Leverages Convolutional Neural Networks to capture spatial and temporal features of EEG signals.  
  2. **Logistic Regression Model** (`LogisticModelForMentalWorkloadClassification.ipynb`)  
     - A baseline linear classifier for performance comparison.  

- Dataset preprocessing and feature extraction were performed using **SciPy** and **NumPy**.  
- Evaluation was done using metrics like **accuracy** and **confusion matrix**, along with **K-Fold cross-validation** for robust results.  

---

## 🛠️ Tech Stack
- **Languages & Libraries**: Python, NumPy, Matplotlib, SciPy  
- **Deep Learning**: TensorFlow, Keras (Sequential API, Conv1D, Conv2D, MaxPooling, Dense, Dropout)  
- **Machine Learning**: Scikit-learn (Logistic Regression, train_test_split, cross_val_score)  
- **Metrics**: Accuracy, Confusion Matrix  

---

## ⚙️ Workflow

### 1. Data Loading & Preprocessing
- EEG/mental workload dataset loaded using `scipy.io`  
- Features normalized and prepared for ML/DL pipelines  
- Split dataset into training and testing sets  

### 2. Logistic Regression Model
- Implemented with **Scikit-learn**  
- Trained using **K-Fold Cross Validation**  
- Evaluated with **accuracy score**  

### 3. Convolutional Neural Network (CNN) Model
- **Architecture**:
  - Convolutional layers (`Conv1D`, `Conv2D`) for feature extraction  
  - **MaxPooling** layers for dimensionality reduction  
  - **Flatten + Dense layers** for classification  
  - **Dropout** to reduce overfitting  
- **Loss Function**: Binary Crossentropy  
- **Optimizer**: Adam  
- **Evaluation**: Accuracy Score & Confusion Matrix  

### 4. Evaluation
- Logistic Regression: Provides interpretability, acts as baseline  
- CNN: Captures deeper patterns in workload data, performs better in classification tasks  

---

## 📊 Results
- **Logistic Regression**: Works well for linearly separable features but limited with complex patterns  
- **CNN**: Achieved higher accuracy and robustness in classifying mental workload  
- Confusion matrix revealed CNN reduces misclassifications compared to Logistic Regression  

---

