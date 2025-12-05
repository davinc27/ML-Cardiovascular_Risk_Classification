# ML-Cardiovascular Risk Classification

This repository provides a complete, step-by-step machine learning workflow for predicting cardiovascular risk. Every aspect is implemented in Jupyter Notebooks for transparency and reproducibility.

## Project Overview

Cardiovascular diseases are among the leading causes of death worldwide. Accurately identifying individuals at risk before symptoms appear enables earlier intervention and improved health outcomes. This repository demonstrates how to preprocess data, explore key relationships, select and engineer features, train and validate machine learning models, and interpret the results using real medical data.

## Dataset

The dataset used for cardiovascular risk classification includes the following:
- **Demographic Data**: age, gender, height, weight
- **Vital Signs**: systolic and diastolic blood pressure
- **Blood Chemistry**: total cholesterol levels, glucose levels
- **Lifestyle Factors**: smoking status, alcohol intake, level of physical activity
- **Medical History**: whether the individual has hypertension, diabetes, or any diagnosed cardiovascular disease

## Code Walkthrough

All code is provided in ordered Jupyter Notebooks, designed to enable users to follow the project workflow from start to finish.

### 1. Data Preprocessing (`01_data_preprocessing.ipynb`)
- **Step 1:** Load raw dataset (CSV, Excel, or DataFrame)
- **Step 2:** Inspect dataset structure and variable types
- **Step 3:** Handle missing values:
    - Remove or impute missing entries using mean/mode
- **Step 4:** Encode categorical variables (e.g., gender, smoking status) using label or one-hot encoding
- **Step 5:** Scale numerical features (e.g., age, cholesterol) with StandardScaler or MinMaxScaler
- **Step 6:** Save the cleaned dataset for use in subsequent notebooks

### 2. Exploratory Data Analysis (`02_eda.ipynb`)
- **Step 1:** Visualize distribution of each feature using histograms and boxplots
- **Step 2:** Analyze relationships between features and the target label using bar charts, scatter plots, and correlation heatmaps
- **Step 3:** Identify outliers and examine potential impacts on model performance
- **Step 4:** Summarize findings to inform feature engineering

### 3. Feature Engineering and Selection (`03_feature_engineering.ipynb`)
- **Step 1:** Create meaningful features, e.g. BMI calculation from height and weight
- **Step 2:** Combine or derive new variables: e.g. risk scores based on clinical guidelines
- **Step 3:** Remove highly correlated or uninformative features
- **Step 4:** Use techniques such as Recursive Feature Elimination (RFE) to select the best predictors
- **Step 5:** Prepare final feature set for modeling

### 4. Model Training and Hyperparameter Tuning (`04_model_training.ipynb`)
- **Step 1:** Split dataset into training and test sets (typically 80/20 split)
- **Step 2:** Select classification algorithms. Common choices include:
    - Logistic Regression
    - Random Forest Classifier
    - Gradient Boosting Classifier
    - Support Vector Machine
- **Step 3:** Train models using training data
- **Step 4:** Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- **Step 5:** Compare models based on validation scores and select the best performer

### 5. Evaluation and Interpretation (`05_evaluation.ipynb`)
- **Step 1:** Evaluate final model on test set using metrics such as:
    - Accuracy: proportion of correct predictions
    - Precision: proportion of true positives among all predicted positives
    - Recall: proportion of true positives among all actual positives
    - F1-score: harmonic mean of precision and recall
    - ROC-AUC: evaluates separability of classes
- **Step 2:** Display confusion matrix and ROC curve plots
- **Step 3:** Interpret feature importance to understand which variables most influence cardiovascular risk
- **Step 4:** Summarize results and discuss limitations or potential improvements

## Getting Started

Follow these instructions to reproduce the analysis or run experiments on your own data.

### 1. Clone the Repository
```bash
git clone https://github.com/davinc27/ML-Cardiovascular_Risk_Classification.git
cd ML-Cardiovascular_Risk_Classification
```

### 2. Set Up the Environment
Ensure you have Python (>=3.7) and Jupyter Notebook installed. Use a virtual environment for best practice.
```bash
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
pip install -r requirements.txt   # Ensure requirements.txt is available
```

### 3. Launch Jupyter Notebook
```bash
jupyter notebook
```
Open notebooks in your browser and run cells sequentially, starting from `01_data_preprocessing.ipynb`.

### 4. Prepare and Use Your Data
- Place your data in the appropriate directory (`/data` recommended)
- Adjust data paths in the notebooks if necessary

## Results

Key findings and performance metrics will be presented in the final notebook (`05_evaluation.ipynb`), including:
- Best model type and tuned parameters
- Test set accuracy, precision, recall, F1-score, and ROC-AUC scores
- Feature importance ranking and insights
- Visualization of confusion matrix and ROC curve

**Sample Results:**
- Random Forest achieved a test accuracy of 0.87, F1-score of 0.85, and ROC-AUC of 0.91
- Most predictive features: age, systolic blood pressure, smoking status, cholesterol

---

**Maintainer:** davinc27
