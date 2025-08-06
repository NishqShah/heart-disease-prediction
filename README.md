# Heart Disease Analysis and Prediction

This machine learning project focuses on analyzing heart disease data and predicting whether a person is likely to have heart disease based on several clinical and lifestyle features. The goal is to develop a reliable prediction model by training and comparing multiple machine learning algorithms.

---

##  Dataset

- **File**: `heart.csv`
- **Total Entries**: 918
- **Features**:
  - `age`: Age of the patient
  - `sex`: Gender (1 = Male, 0 = Female)
  - `cp`: Chest Pain Type (0–3)
  - `trestbps`: Resting Blood Pressure
  - `chol`: Serum Cholesterol
  - `fbs`: Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)
  - `restecg`: Resting ECG results (0–2)
  - `thalach`: Maximum heart rate achieved
  - `exang`: Exercise-induced angina (1 = yes; 0 = no)
  - `oldpeak`: ST depression induced by exercise
  - `slope`: Slope of the peak exercise ST segment
  - `ca`: Number of major vessels (0–3) colored by fluoroscopy
  - `thal`: Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)
  - `target`: 0 = No Heart Disease, 1 = Heart Disease

---

##  Data Preprocessing

- Removed **duplicate** entries
- Checked for **null values** (none found)
- Applied **StandardScaler** to continuous features for normalization
- Split dataset into **training and test sets** (80/20)

---

##  Exploratory Data Analysis (EDA)

Performed several visualizations (available in the notebook) to understand feature distributions and relationships.

- Gender distribution by heart disease status
- Chest pain type vs target variable
- Fasting blood sugar vs target
- Resting blood pressure and cholesterol histograms
- KDE plots by gender for blood pressure
- Correlation heatmap of all features

---

##  Machine Learning Models Used

The following classification models were trained and evaluated:

| Model                     | Accuracy |
|---------------------------|----------|
| Logistic Regression       | 78.6%    |
| K-Nearest Neighbors       | 78.6%    |
| Support Vector Machine    | 80.3%    |
| Decision Tree             | 77.0%    |
| Random Forest             | 86.8%    | *(Best)*
| Gradient Boosting         | 81.9%    |
The **Random Forest Classifier** showed the best performance and was selected for final use.

---

##  Model Evaluation

- All models were evaluated using the **accuracy score**
- Compared results using a **bar plot**
- Selected the **Random Forest Classifier** based on highest accuracy

---

##  Final Results

- **Best Model**: Random Forest Classifier
- **Accuracy**: 86.8%
- **Confusion Matrix** and **Classification Report**: Included in notebook

---

##  Final Model

- Trained the final **Random Forest Classifier** on the complete dataset
- Used for demonstration predictions on new sample inputs
---

##  Model Deployment

- Final model saved using `joblib`:
  - `model_joblib_heart_disease.pkl`
- Sample prediction workflow included:
  - Load model
  - Input new data as `DataFrame`
  - Predict and interpret result ("Disease" or "No Disease")

---

##  Files in This Repository

- `heart_disease_prediction.ipynb` — Full code including EDA, model training, and prediction
- `heart.csv` — The dataset used
- `model_joblib_heart_disease.pkl` — Final saved model

---
