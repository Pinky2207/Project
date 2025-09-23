# 🩺 Diabetes Prediction using Machine Learning

This project applies **machine learning regression models** to predict diabetes progression in patients based on medical and lifestyle features.  
The workflow includes exploratory data analysis (EDA), feature preprocessing, model training, evaluation, and comparison of different regression algorithms.  

---

## 📘 Project Overview
The objective of this project is to:
- Explore and understand patient health data using **EDA**  
- Preprocess categorical and numerical features for ML  
- Train multiple regression models to predict diabetes outcomes  
- Evaluate models with multiple regression metrics  
- Compare strengths and weaknesses of each algorithm  

---

## 🛠️ Tech Stack
- **Languages & Libraries**: Python, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn  
- **Preprocessing**: OneHotEncoder, ColumnTransformer, StandardScaler  
- **Machine Learning Models**:
  - Linear Regression  
  - Decision Tree Regressor  
  - Random Forest Regressor  
  - Gradient Boosting Regressor  
- **Evaluation Metrics**: MAE, RMSE, MSE, R² Score, Cross-Validation (K-Folds)  

---

## ⚙️ Workflow

1. **Exploratory Data Analysis (EDA)**  
   - Analyzed missing values, feature distributions, and correlations  
   - Visualized relationships between features and diabetes outcome  

2. **Data Preprocessing**  
   - Encoded categorical variables using **OneHotEncoder**  
   - Used **ColumnTransformer** to handle mixed data types  
   - Applied **StandardScaler** for numerical feature scaling  

3. **Model Training & Testing**  
   - Split dataset into training and test sets  
   - Trained multiple regression models (Linear, Decision Tree, Random Forest, Gradient Boosting)  

4. **Evaluation**  
   - Evaluated performance using:  
     - Mean Absolute Error (MAE)  
     - Root Mean Absolute Error (RMAE)  
     - Mean Squared Error (MSE)  
     - Root Mean Squared Error (RMSE)  
     - R² Score  
   - Performed **Cross-Validation with K-Folds** for robust model evaluation  

5. **Model Insights**  
   - Compared models to understand strengths and weaknesses  
   - Identified Random Forest and Gradient Boosting as stronger performers due to their ability to capture non-linear relationships  
   - Noted limitations of Linear Regression with complex feature interactions  

---


