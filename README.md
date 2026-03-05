# Autism Prediction using Machine Learning

A Machine Learning project that predicts **Autism Spectrum Disorder (ASD)** using screening questionnaire scores and demographic information.  
This project demonstrates a complete **end-to-end ML workflow** including data preprocessing, exploratory data analysis (EDA), handling imbalanced data, model training, hyperparameter tuning, evaluation, and saving the trained model.

# Project Overview

Autism Spectrum Disorder (ASD) is a developmental condition that affects communication and behavior. Early detection can help in timely intervention.

This project builds a **machine learning model** that predicts whether a person is likely to have **ASD or not** based on questionnaire scores and personal information.

The notebook implements:

- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Feature encoding
- Outlier detection and treatment
- Handling class imbalance using **SMOTE**
- Training multiple ML models
- Hyperparameter tuning
- Model evaluation
- Saving trained models for reuse

---

# Repository Structure

```
Autism-Prediction-ML/
│
├── Autism_Preidiction_using_machine_Learning.ipynb   # Main project notebook
├── train.csv                                         # Dataset
├── best_model.pkl                                    # Saved trained model
├── encoders.pkl                                      # Saved label encoders
├── requirements.txt                                  # Project dependencies
└── README.md                                         # Project documentation
```

---

# Dataset Information

The dataset contains screening questions and demographic attributes.

### Target Variable

| Column | Description |
|------|-------------|
| `Class/ASD` | Indicates whether a person has Autism (1) or not (0) |

### Example Feature Columns

- `A1_Score` – `A10_Score` → Screening questionnaire responses
- `age` → Age of the person
- `gender`
- `ethnicity`
- `jaundice`
- `austim`
- `country_of_res`
- `used_app_before`
- `relation`
- `result`

Some categorical columns contain `"?"` values which are treated as **unknown values** and mapped to `"Others"` during preprocessing.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/autism-prediction-ml.git
cd autism-prediction-ml
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

# Requirements

Create a file named **requirements.txt**

```
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
joblib
```

Install them using:

```bash
pip install -r requirements.txt
```

---

# How to Run the Project

## Option 1: Using Jupyter Notebook (Local)

1. Place `train.csv` in the project directory
2. Run Jupyter Notebook

```bash
jupyter notebook
```

3. Open

```
Autism_Preidiction_using_machine_Learning.ipynb
```

4. Run all cells sequentially.

---

## Option 2: Using Google Colab

1. Upload the notebook to **Google Colab**
2. Upload `train.csv`
3. Change dataset path if needed:

```python
df = pd.read_csv("/content/train.csv")
```

4. Run all cells.

---

# Data Preprocessing

The following preprocessing steps are applied:

### Handling Missing / Unknown Values

Some columns contain `"?"`.

These values are replaced with `"Others"`.

Example columns:

- `ethnicity`
- `relation`

---

### Label Encoding

Categorical variables are converted into numerical form using **LabelEncoder**.

Encoders are saved into:

```
encoders.pkl
```

This allows consistent preprocessing during inference.

---

### Outlier Detection

Outliers are detected using the **IQR (Interquartile Range) method**.

Columns checked:

- `age`
- `result`

Outliers are replaced with the **median value**.

---

### Handling Imbalanced Data

The dataset contains **class imbalance**.

To address this, **SMOTE (Synthetic Minority Oversampling Technique)** is applied to the **training data only**.

Library used:

```
imbalanced-learn
```

---

# Machine Learning Models

The notebook trains and compares several ML algorithms:

### Decision Tree

```
DecisionTreeClassifier
```

---

### Random Forest

```
RandomForestClassifier
```

---

### XGBoost

```
XGBClassifier
```

---

# Hyperparameter Tuning

Hyperparameters are optimized using:

```
RandomizedSearchCV
```

This helps find the best model configuration.

Example tuned parameters:

- `n_estimators`
- `max_depth`
- `bootstrap`
- `learning_rate`
- `subsample`

---

# Model Evaluation

Models are evaluated using:

- Accuracy Score
- Confusion Matrix
- Classification Report
- Cross Validation

Example metrics generated:

```
Accuracy Score
Precision
Recall
F1 Score
```

---

# Best Model

After hyperparameter tuning, the best performing model was:

```
RandomForestClassifier
```

Example configuration:

```
RandomForestClassifier(
    bootstrap=False,
    max_depth=20,
    n_estimators=50,
    random_state=42
)
```

Example performance:

| Metric | Value |
|------|------|
| Cross Validation Accuracy | ~93% |
| Test Accuracy | ~82% |

*(Results may vary depending on dataset splits.)*


# Saved Artifacts

After training, the notebook saves the following files:

### Model File

```
best_model.pkl
```

Contains the trained ML model.

---

### Encoder File

```
encoders.pkl
```

Contains LabelEncoders used during preprocessing.

These are required when making predictions on new data.


#  Prediction Script Example

Example code for loading the model and predicting:

```python
import joblib
import pandas as pd

# Load model and encoders
model = joblib.load("best_model.pkl")
encoders = joblib.load("encoders.pkl")

# Example input
sample = pd.DataFrame({
    "age":[25],
    "gender":["Male"]
})

# Encode categorical columns
for col, encoder in encoders.items():
    sample[col] = encoder.transform(sample[col])

prediction = model.predict(sample)

print("Prediction:", prediction)
```

# Exploratory Data Analysis

The notebook includes visualizations such as:

- Feature distributions
- Correlation heatmap
- Class imbalance visualization
- Outlier analysis
- Feature importance

Libraries used:

```
Matplotlib
Seaborn
```

#  Future Improvements

Possible enhancements for the project:

- Build a **Flask / FastAPI web app**
- Deploy the model using **Docker**
- Add **SHAP explainability**
- Use **Pipeline + ColumnTransformer**
- Create **REST API for predictions**
- Deploy on **AWS / Render / HuggingFace Spaces**

#  Technologies Used

| Category | Tools |
|------|------|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Imbalanced Learning | imbalanced-learn |
| Gradient Boosting | XGBoost |
| Model Saving | Joblib |
| Development | Jupyter Notebook |


