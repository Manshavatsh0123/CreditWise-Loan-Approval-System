# CreditWise Loan Approval System

An end-to-end Machine Learning project that predicts whether a loan application should be **Approved** or **Rejected** using supervised learning algorithms. The project follows a complete ML pipeline, including data preprocessing, feature engineering, model training, and evaluation to build an accurate binary classification system.

---

## Project Overview

CreditWise automates the loan approval prediction process by analysing applicant information and predicting loan eligibility.

This project demonstrates the complete machine learning workflow:

- Data Cleaning
- Missing Value Handling
- Exploratory Data Analysis (EDA)
- Feature Encoding
- Correlation Analysis
- Feature Engineering
- Feature Scaling
- Model Training
- Model Evaluation

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

## Project Workflow

```
Dataset
   │
   ▼
Data Cleaning
   │
   ▼
Missing Value Handling
   │
   ▼
Exploratory Data Analysis
   │
   ▼
Feature Encoding
   │
   ▼
Correlation Analysis
   │
   ▼
Feature Engineering
   │
   ▼
Train-Test Split
   │
   ▼
Feature Scaling
   │
   ▼
Model Training
   │
   ▼
Prediction
   │
   ▼
Model Evaluation
```

---

## Feature Engineering

To improve model performance, new features were created from existing variables.

- Squared DTI Ratio
- Squared Credit Score
- Log-transformed Applicant Income

Example:

```python
creditWiseData["DTI_Ratio_sq"] = creditWiseData["DTI_Ratio"] ** 2
creditWiseData["Credit_Score_sq"] = creditWiseData["Credit_Score"] ** 2
creditWiseData["Applicant_Income_log"] = np.log1p(
    creditWiseData["Applicant_Income"]
)
```

---

## Machine Learning Models

The following supervised learning algorithms were implemented:

- Logistic Regression
- K-Nearest Neighbours (KNN)
- Naive Bayes

---

## Model Evaluation

The models were evaluated using standard classification metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## Key Skills Demonstrated

- Data Preprocessing
- Feature Engineering
- Exploratory Data Analysis
- Feature Scaling
- Binary Classification
- Machine Learning
- Model Evaluation
- Data Visualisation
- Scikit-learn

---

## Project Structure

```
CreditWise-Loan-Approval-System/
│
├── CreditWise_Loan_System.ipynb
├── dataset/
├── README.md
└── requirements.txt
```

---

## Getting Started

Clone the repository

```bash
git clone https://github.com/Manshavatsh0123/CreditWise-Loan-Approval-System.git
```

Move into the project directory

```bash
cd CreditWise-Loan-Approval-System
```

Install the required dependencies

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook

```bash
jupyter notebook
```

---

## Future Enhancements

- Hyperparameter Tuning
- Cross Validation
- Random Forest
- XGBoost
- LightGBM
- Model Deployment using FastAPI or Flask
- Streamlit Dashboard

---

## Author

**Mansha Vatsh**

Software Developer | Machine Learning Enthusiast

**GitHub**  
https://github.com/Manshavatsh0123

**LinkedIn**  
https://www.linkedin.com/in/mansha-vatsh
