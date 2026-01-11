# Insurance Premium Prediction Project
## A comprehensive machine learning project for predicting insurance premiums using multiple regression models and advanced feature engineering techniques.


## 🎯 Overview
This project implements an end-to-end machine learning pipeline to predict insurance premiums based on customer demographics, policy details, and behavioral factors. The solution explores multiple regression algorithms, feature engineering techniques, and dimensionality reduction methods to achieve optimal prediction accuracy.
✨ Features

Comprehensive Data Analysis: Exploratory data analysis with interactive Plotly visualizations
Multiple ML Models: Implementation of Linear Regression, Ridge, Lasso, Random Forest, and Gradient Boosting
Feature Engineering: Creation of interaction terms and temporal features
Dimensionality Reduction: PCA analysis for feature optimization
Hyperparameter Tuning: GridSearchCV for model optimization
Data Preprocessing: Advanced handling of missing values and categorical encoding
Interactive Visualizations: Dynamic charts for better insights

## 📊 Dataset
The dataset contains 200,000 training samples and 100,000 test samples with the following features:
Demographic Features

Age, Gender, Marital Status
Number of Dependents
Education Level, Occupation

Financial Features

Annual Income
Credit Score
Premium Amount (Target Variable)

Policy Features

Policy Type (Basic, Comprehensive, Premium)
Policy Start Date
Insurance Duration
Previous Claims

Health & Lifestyle Features

Health Score
Smoking Status
Exercise Frequency

Property Features

Location (Urban, Suburban, Rural)
Property Type (House, Apartment, Condo)

## 🚀 Installation
Prerequisites
bashPython 3.8+
pip
Setup
bash# Clone the repository
git clone https://github.com/yourusername/insurance-premium-prediction.git
cd insurance-premium-prediction

## Create virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

## 🚀 Install Dependencies

### Install the required Python packages using:

```bash
pip install -r requirements.txt
Required Libraries
Library	Version
numpy	1.26.4
pandas	1.5.3
plotly	5.9.0
scikit-learn	1.3.0
scipy	1.10.1
seaborn	0.12.2
matplotlib	3.7.1
missingno	0.5.2
lightgbm	4.6.0
statsmodels	0.13.5

📁 Project Structure
bash
Copy code
insurance-premium-prediction/
│
├── DS_Project.ipynb          # Main Jupyter notebook
├── requirements.txt          # Project dependencies
├── README.md                 # Project documentation
🔬 Methodology
1️⃣ Data Preprocessing
Missing Value Imputation:

Numerical → Mean

Categorical → Mode

Date Feature Engineering: Extracted year, month, and day from Policy Start Date

Temporal Features: Created Days Since Policy Start and Years Since Policy Start

Duplicate Removal: Verified no duplicate records

2️⃣ Exploratory Data Analysis (EDA)
Distribution analysis of numerical features

Correlation analysis between features

Categorical feature relationships with target variable

Interactive visualizations using Plotly

3️⃣ Feature Engineering
Interaction Terms:

Income_Per_Dependent = Annual Income / (Number of Dependents + 1)

Claims_Ratio = Previous Claims / (Insurance Duration + 1)

Mutual Information Analysis: Identify the most informative categorical features

4️⃣ Model Development
Implemented and compared multiple models:

Linear Regression (Baseline)

Ridge Regression (L2 Regularization)

Lasso Regression (L1 Regularization)

Random Forest Regressor

Gradient Boosting Regressor

5️⃣ Dimensionality Reduction
PCA analysis with 15 components

Variance explained analysis

Model performance evaluation with reduced features
## 📈 Models & Results

### Model Performance Comparison

| Model                 | MAE          | MSE         | R² Score      | Notes                        |
|-----------------------|-------------|------------|---------------|-------------------------------|
| Linear Regression     | 1.13e-12    | 2.17e-24   | 1.0           | Best baseline performance     |
| Ridge (α=0.01)        | 1.66e-04    | 4.55e-08   | 1.0           | Excellent with regularization |
| Lasso (α=0.01)        | 7.67e-03    | 9.71e-05   | 1.0           | Good feature selection        |
| Random Forest         | 0.065       | 0.244      | 0.99999997    | Robust to outliers            |
| Gradient Boosting     | 2.538       | 13.374     | 0.9999815     | Good generalization           |

---

### Best Hyperparameters

**Random Forest**
- `n_estimators`: 200  
- `max_depth`: 30  
- `min_samples_split`: 2  
- `min_samples_leaf`: 1  

**Gradient Boosting**
- `n_estimators`: 200  
- `learning_rate`: 0.05  
- `max_depth`: 5  

---

### PCA Results

| Components | Variance Explained |
|------------|------------------|
| 11         | 90%              |
| 15         | >99%             |

**Linear Regression with 11 PCA components**  
- **MAE**: 5.40e-05  
- **R²**: 0.9999999999999919




## 📊 Visualizations
The project includes comprehensive visualizations:

Distribution Plots

Premium Amount distribution by Policy Type
Age distribution across demographics
Income distribution by location


Relationship Analysis

Scatter plots: Age vs Premium Amount
Box plots: Premium by Gender, Location, Property Type
Correlation heatmaps


Time Series Analysis

Premium trends over Policy Years
Seasonal patterns in policy enrollment


Feature Importance

Random Forest feature importance
Mutual Information scores for categorical features


Model Comparison

Interactive bar charts comparing MAE, MSE, R²
Residual plots for model diagnostics



## 💡 Key Insights

Feature Importance:

Policy Type significantly impacts premium
Exercise Frequency and Location show strong relationships
Education Level correlates with premium amounts


Data Quality:

~30% missing values in Occupation
~15% missing values in Previous Claims
Successfully handled through strategic imputation


Model Selection:

Linear models perform excellently with proper feature engineering
Tree-based models provide robustness to outliers
Regularization prevents overfitting


Feature Engineering Impact:

Interaction terms improve model interpretability
Temporal features capture policy lifecycle effects
PCA reduces dimensionality while maintaining 90% variance



## 🛠️ Technologies Used

Python 3.11: Core programming language
Pandas: Data manipulation and analysis
NumPy: Numerical computations
Scikit-learn: Machine learning models and preprocessing
Plotly: Interactive visualizations
Matplotlib/Seaborn: Static visualizations
LightGBM: Gradient boosting framework
Statsmodels: Statistical modeling

## 💻 Usage
Running the Notebook
bashjupyter notebook DS_Project.ipynb
Training Models
python# Load and preprocess data
train_data = pd.read_csv('data/train_sample.csv')
train_sampled = train_data.sample(n=50000)

### Train Random Forest
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=200, max_depth=30)
rf_model.fit(X_train_scaled, y_train)

### Make predictions
predictions = rf_model.predict(X_test_scaled)
Generating Visualizations
pythonimport plotly.express as px

### Distribution plot
fig = px.histogram(train_sampled, x='Premium Amount', 
                   color='Policy Type',
                   title='Premium Distribution by Policy Type')
fig.show()
## 🤝 Contributing
Contributions are welcome! Please follow these steps:

Fork the repository
Create a feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request

Development Guidelines

Follow PEP 8 style guidelines
Add docstrings to all functions
Include unit tests for new features
Update README.md with new features

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
## 👥 Author

marwan eslam ouda

