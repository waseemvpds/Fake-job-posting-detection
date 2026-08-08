# 🛡️ Fraudulent Job Posting Detection System

A machine-learning based web application that analyzes job advertisements and estimates whether a posting is **fraudulent or legitimate**.

The project combines:

- Natural Language Processing (NLP) using **TF-IDF**
- Categorical feature encoding
- Numerical and binary feature engineering
- **LightGBM** classification
- Randomized hyperparameter tuning
- Input-quality and gibberish detection
- Rule-based fraud-risk signals
- Interactive Streamlit interface
- Batch prediction
- Side-by-side job comparison
- Real-or-Fake interactive game
- Dataset analytics
- Downloadable PDF risk reports

> **Important:** The system is a decision-support tool. Its prediction is not a legal determination and should not be treated as proof that a job posting is fraudulent.

---

## 📌 Project Overview

Fake job postings are a common problem on online recruitment platforms. Fraudulent listings can contain misleading descriptions, unrealistic promises, requests for money or personal information, and incomplete company information.

This project uses the **Fake Job Postings** dataset to train a binary classification model that predicts the `fraudulent` target.

The trained model analyzes both the **textual content** and **structured metadata** of a job posting.

The final application is implemented as a single-file Streamlit application.

---

## ✨ Main Features

### 🔍 Single Analysis

Analyze one job posting by entering:

- Job title
- Company profile
- Job description
- Requirements
- Benefits
- Employment type
- Required experience
- Required education
- Industry
- Function
- Country
- State
- City
- Telecommuting status
- Company logo availability
- Screening questions
- Salary availability

The application returns:

- Fraudulent / Legitimate verdict
- Fraud probability
- Risk level
- Risk indicators
- Model-related signals
- Recommendations
- Detailed assessment information

---

### 📦 Batch Analysis

Upload a CSV containing multiple job postings and run predictions for many listings at once.

Expected fields include:

```text
title
company_profile
description
requirements
benefits
employment_type
required_experience
required_education
industry
function
country
state
city
telecommuting
has_company_logo
has_questions
has_salary
```

Missing fields can be handled by the application as blank or `Unknown` values where appropriate.

---

### ⚖️ Compare Postings

The application provides a comparison interface where two job advertisements can be analyzed independently and compared side by side.

The comparison workflow includes:

1. Parsing the pasted job advertisements
2. Extracting the main text sections
3. Checking input quality
4. Running the trained model
5. Calculating fraud-risk signals
6. Displaying the results for both postings

---

### 🎮 Real or Fake?

An interactive game is included where users can examine job postings and predict whether they are real or fraudulent before revealing the AI assessment.

The game supports difficulty levels and maintains session-based scoring and streak information.

---

### 📊 Dataset Statistics

The application can load the original CSV dataset and display analytical information such as:

- Dataset size
- Fraudulent vs legitimate postings
- Employment-type distributions
- Industry distributions
- Function distributions
- Country distributions
- Fraud-related patterns
- Suspicious phrase analysis

---

### 🧪 Input Quality Gate

Before a job posting reaches the prediction model, the application performs several quality checks.

The quality gate checks for issues such as:

- Empty required fields
- Very short text
- Too few words
- Implausible word lengths
- Random or gibberish text
- Unnatural vowel distribution
- Unpronounceable words
- Excessive repeated characters
- Excessive punctuation
- Excessive numeric content
- Repeated sentences
- Abnormal character entropy
- Unusual English letter patterns
- Extremely sparse submissions

This helps prevent obviously invalid input from being sent directly to the model.

---

### 🚨 Fraud Risk Rules

In addition to the machine-learning model, the application contains transparent rule-based risk signals.

Examples include:

- Registration or processing fees
- Security deposits
- Requests for bank details
- Requests for card or identity information
- Wire transfers
- Cryptocurrency or gift-card payments
- WhatsApp / Telegram recruitment
- Guaranteed income
- Unrealistic earnings
- Artificial urgency
- "No experience required" scam patterns
- External short links
- Suspicious low-skill job formats

These rules are used to provide additional explainability and risk context.

---

### 📄 PDF Risk Report

The application can generate a formal PDF report containing information such as:

- Report identifier
- Generation timestamp
- Job title
- Location
- Prediction outcome
- Fraud probability
- Risk level
- Input quality
- Risk assessment signals
- Submitted job details
- Company profile
- Job description
- Requirements
- Benefits

The PDF is generated using **ReportLab**.

---

## 🧠 Machine Learning Pipeline

The notebook develops the model using a preprocessing and classification pipeline.

### Dataset

The project uses the **Fake Job Postings** dataset.

Dataset shape used in the notebook:

```text
17,880 rows
18 columns
```

The target variable is:

```text
fraudulent
```

where:

```text
0 = Legitimate
1 = Fraudulent
```

---

## 🧹 Data Preprocessing

The notebook performs several preprocessing operations.

### Missing Text Values

The following text fields are converted to empty strings when missing:

```text
title
company_profile
description
requirements
benefits
```

### Categorical Values

Categorical fields are filled with:

```text
Unknown
```

when values are missing.

The main categorical features are:

```text
employment_type
required_experience
required_education
industry
function
country
state
city
```

### Location Processing

The original `location` field is split into:

```text
country
state
city
```

---

## 🛠️ Feature Engineering

The final model uses several groups of features.

### Text Feature

A combined text field is created from:

```text
title
company_profile
description
requirements
benefits
```

The combined text is cleaned and transformed using **TF-IDF**.

### Numerical Features

The model uses:

```text
description_length
requirements_length
company_profile_length
benefits_length
suspicious_word_count
```

### Binary Features

The model uses:

```text
telecommuting
has_company_logo
has_questions
has_salary
has_company_profile
has_requirements
```

### Categorical Features

The model uses:

```text
employment_type
required_experience
required_education
industry
function
country
state
city
```

The Streamlit application reproduces the same feature construction used by the notebook so that the saved pipeline receives the expected input structure.

---

## 🧹 Text Cleaning

The application and notebook perform text cleaning before TF-IDF processing.

The cleaning process includes:

- HTML entity decoding
- Removal of URL placeholders
- Removal of email placeholders
- Removal of phone placeholders
- Lowercasing
- Removal of punctuation
- Removal of isolated single letters
- Whitespace normalization

---

## 🤖 Model Development

Several classification algorithms were evaluated in the notebook, including:

- Logistic Regression
- Linear SVM
- Multinomial Naive Bayes
- Random Forest
- XGBoost
- LightGBM
- CatBoost

LightGBM was selected for the final pipeline.

The final architecture is conceptually:

```text
Raw Job Posting
       │
       ▼
Data Cleaning
       │
       ├── Text → TF-IDF
       ├── Categorical → One-Hot Encoding
       ├── Numerical → Passthrough
       └── Binary → Passthrough
       │
       ▼
ColumnTransformer
       │
       ▼
LightGBM Classifier
       │
       ▼
Fraud Probability
       │
       ▼
Fraudulent / Legitimate
```

---

## 🎯 Hyperparameter Tuning

The final LightGBM pipeline was tuned using:

```text
RandomizedSearchCV
```

The notebook searches across parameters including:

- `n_estimators`
- `learning_rate`
- `num_leaves`
- `max_depth`
- `min_child_samples`
- `subsample`
- `colsample_bytree`
- `reg_alpha`
- `reg_lambda`

The search uses:

```text
n_iter = 10
scoring = F1
cv = 3
random_state = 42
```

The notebook records a best cross-validation F1 score of approximately:

```text
0.8259
```

---

## 📈 Model Evaluation

The notebook evaluates the model using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- Confusion Matrix
- ROC Curve
- Precision-Recall analysis

The notebook contains multiple evaluation outputs from different stages of model development. Therefore, the application displays its own configured metric values rather than assuming every notebook output represents the currently deployed model.

The metrics configured in the current Streamlit application are:

| Metric | Value |
|---|---:|
| Accuracy | 98.46% |
| Precision | 90.68% |
| Recall | 75.61% |
| F1 Score | 82.46% |
| ROC AUC | 99.05% |

These values are the metrics currently presented by the application UI.

---

## 📦 Project Files

A typical project directory should contain:

```text
Fraudulent-Job-Posting-Detection/
│
├── app.py
├── 2nd(4).ipynb
├── fjd_model.pkl
├── fake_job_postings.csv
├── README.md
└── requirements.txt
```

### `app.py`

Main Streamlit application.

It contains:

- User interface
- Model loading
- Feature engineering
- Input validation
- Risk rules
- Single prediction
- Batch prediction
- Comparison
- Interactive game
- Dataset statistics
- PDF report generation

### `2nd(4).ipynb`

Model-development notebook containing:

- Data loading
- Data exploration
- Data cleaning
- Feature engineering
- Model comparison
- Cross-validation
- LightGBM tuning
- Evaluation
- Error analysis
- Model serialization

### `fjd_model.pkl`

Serialized trained scikit-learn pipeline containing the preprocessing and LightGBM model.

The notebook saves the trained model using:

```python
pickle.dump(best_model, open("fjd_model.pkl", "wb"))
```

### `fake_job_postings.csv`

Original dataset used for model development and, when available, to enrich application dropdown options and dataset analytics.

The dataset is optional for basic prediction because the application contains fallback options, but it is required for the full dataset-statistics and dataset-enrichment functionality.

---

## ⚙️ Installation

### 1. Clone or download the project

Place the project files in the same directory.

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Streamlit Application

From the project directory:

```bash
streamlit run app.py
```

Streamlit will provide a local URL, normally similar to:

```text
http://localhost:8501
```

---

## 🧠 Running the Notebook

If you want to reproduce the model-development workflow:

```bash
jupyter notebook
```

Open:

```text
2nd(4).ipynb
```

The notebook loads:

```text
fake_job_postings.csv
```

and ultimately saves:

```text
fjd_model.pkl
```

Make sure the generated model is placed in the same directory as `app.py`.

---

## 🔐 Model and Data Requirements

The application expects:

```text
fjd_model.pkl
```

in the application directory.

The dataset path configured in the application is:

```text
fake_job_postings.csv
```

The application loads the saved pipeline with Python's `pickle` module.

Because the model is serialized, the environment should contain compatible versions of the libraries used when the model was trained, especially:

```text
scikit-learn
lightgbm
```

For this reason, avoid changing major library versions after creating the model unless the model is retrained and re-exported.

---

## 📁 Batch CSV Format

For batch prediction, a CSV can contain the following columns:

```text
title
company_profile
description
requirements
benefits
employment_type
required_experience
required_education
industry
function
country
state
city
telecommuting
has_company_logo
has_questions
has_salary
```

Boolean fields should preferably be represented as:

```text
0 / 1
```

or values that can be converted into the expected binary representation.

---

## 🧮 Prediction Threshold

The application uses a configured fraud decision threshold of:

```text
0.35
```

The risk bands are based on the configured threshold.

The application also combines the machine-learning probability with transparent rule-based fraud signals for its final risk assessment.

Therefore:

```text
Model probability
        +
Rule-based risk signals
        ↓
Final risk assessment
```

The application is designed to provide both a prediction and an explanation rather than only returning a binary class.

---

## 🔎 Explainability

The application provides human-readable risk factors based on:

- Suspicious phrases
- Missing company information
- Missing requirements
- Missing screening questions
- Missing salary information
- Short descriptions
- Unspecified education
- Unspecified industry
- Unspecified location
- Presence of company logo
- Detailed company profile
- Comprehensive job description
- Screening questions
- Absence of monitored scam phrases

These signals help users understand why a posting may appear risky.

---

## 🛡️ Limitations

This project has several important limitations.

### Dataset limitation

The model learns patterns from the training dataset and may not generalize perfectly to every recruitment platform or geographic region.

### False positives

A legitimate job posting may be classified as suspicious.

### False negatives

A fraudulent posting may appear legitimate.

### Rule-based signals

Suspicious phrases are indicators, not proof of fraud.

For example, legitimate companies may sometimes use terms such as "immediate joining" or recruit through messaging platforms.

### No external verification

The system does not independently verify:

- Company registration
- Recruiter identity
- Domain ownership
- Email authenticity
- Salary authenticity
- Job-board account ownership

Therefore, users should use the result as a screening aid rather than a final judgment.

---

## 🚀 Future Improvements

Possible future improvements include:

- Transformer-based NLP models
- BERT-based text classification
- Better class-imbalance handling
- Probability calibration
- SHAP-based model explanations
- External company verification
- Domain and URL reputation checks
- Recruiter verification
- More extensive multilingual support
- Continuous model monitoring
- Automated model retraining
- Cloud deployment
- User authentication
- Database-backed prediction history

---

## 🧰 Technology Stack

### Programming

- Python

### Machine Learning

- Scikit-learn
- LightGBM
- XGBoost
- CatBoost

### NLP

- TF-IDF
- Text preprocessing
- Suspicious phrase detection

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Web Application

- Streamlit

### Reporting

- ReportLab

### Model Serialization

- Pickle

### Development

- Jupyter Notebook

---

## 👤 Author

**Waseem V P**

- LinkedIn: https://www.linkedin.com/in/waseemvpds
- GitHub: https://github.com/waseemvpds

---

## ⚠️ Disclaimer

This application provides automated machine-learning based risk assessment for informational and decision-support purposes only.

A prediction of "Fraudulent" does not prove that a job advertisement is fraudulent, and a prediction of "Legitimate" does not guarantee that a job advertisement is safe.

Users should independently verify employers, recruiters, contact information, payment requests, and job-board listings before sharing sensitive information or accepting employment.

---

## 📄 License

This project is intended for educational, portfolio, and demonstration purposes.
