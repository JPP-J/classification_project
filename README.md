# Classification Project
![Last Commit](https://img.shields.io/github/last-commit/JPP-J/classification_project?style=flat-square)
![Python](https://img.shields.io/badge/Python-96.2%25-blue?style=flat-square)
![Languages](https://img.shields.io/github/languages/count/JPP-J/classification_project?style=flat-square)

This repo is home to the code that accompanies Jidapa's *Classification Project*. Compares multiple machine learning classification models on a [**bank marketing dataset**](https://drive.google.com/file/d/1QctGSSR5wSQk6cbdjBrKjYcUPs6PHNHN/view?usp=drive_link) with [detail of dataset](https://drive.google.com/file/d/1K2wneqZNolblPX2WBbC2keoxZSg7ATYe/view?usp=sharing)) to identify the best-performing algorithm. The dataset includes client information and marketing campaign outcomes, aiming to predict whether a client will subscribe to a term deposit.

## 📌 Overview

This project compares multiple machine learning classification models to predict whether a client will subscribe to a term deposit based on a **bank marketing dataset** containing client demographics and campaign outcomes.

### 🧩 Problem Statement

Financial institutions conduct marketing campaigns to promote term deposits, but identifying which clients will subscribe is challenging. Using client data and campaign details, the goal is to build classification models that predict the target variable `y` (whether the client subscribed: `yes` or `no`), helping optimize marketing efforts.

### 🔍 Approach

Eight popular classification algorithms are evaluated, including Logistic Regression, Decision Tree, K-Nearest Neighbors, Support Vector Machine, Random Forest, Naive Bayes, Gradient Boosting, and AdaBoost. Models are trained and tested on preprocessed client data featuring demographic and financial attributes.

### 🎢 Processes

1. **Data Loading & Preprocessing** – Load bank marketing dataset, handle missing values (e.g., `age`), encode categorical variables  
2. **Exploratory Data Analysis (EDA)** – Preview dataset and compute descriptive statistics  
3. **Model Training** – Train each classification model on the training set  
4. **Evaluation** – Measure accuracy and confusion matrices for all models with train and test split
5. **Visualization** – Plot decision tree diagrams for interpretability and confusion matrix 
6. **Reporting** – Generate classification reports and compare model performances  

### 🎯 Results & Impact
- ✅ Model Performance Summary [`classifiers.py`](classifiers.py)
- **Best Accuracy:** Logistic Regression and AdaBoost achieved **90% accuracy**  
- **Key Insight:** All models struggle with low recall on the positive class (`yes`), indicating challenges in identifying subscribers reliably  
- **Outcome:** Provides a practical baseline and comparison for marketing campaign classification, enabling more informed targeting strategies

  | Model                  | Accuracy | Confusion Matrix (TN, FP, FN, TP) |
  |------------------------|----------|------------------------------------|
  | Logistic Regression    | 0.90     | [[1187, 11], [128, 27]]            |
  | Decision Tree          | 0.89     | [[1185, 13], [130, 25]]            |
  | K-Nearest Neighbors    | 0.88     | [[1184, 14], [143, 12]]            |
  | Support Vector Machine | 0.89     | [[1198, 0], [155, 0]]              |
  | Random Forest          | 0.88     | [[1172, 26], [130, 25]]            |
  | Naive Bayes            | 0.82     | [[1044, 154], [88, 67]]            |
  | Gradient Boosting      | 0.89     | [[1182, 16], [134, 21]]            |
  | AdaBoost               | 0.90     | [[1184, 14], [126, 29]]            |


### ⚙️ Model Development Challenges

- **Class Imbalance:** Positive subscription cases are fewer, causing low recall in models  
- **Feature Encoding:** Proper handling of categorical features was critical for model effectiveness  
- **Model Selection:** Comparing diverse classifiers helped identify trade-offs between accuracy and recall  
- **Interpretability:** Decision tree visualization aided understanding of key decision rules

## 📂 Project Files

- [`classifiers.py`](classifiers.py): Trains and evaluates the following classification models:
  - Logistic Regression
  - Decision Tree
  - K-Nearest Neighbors
  - Support Vector Machine
  - Random Forest
  - Naive Bayes
  - Gradient Boosting
  - AdaBoost

  Each model reports its accuracy and confusion matrix.

- [`main.py`](main.py)
  - Loads and preprocesses the bank dataset
  - Displays a preview and dataset statistics
  - Trains and evaluates Logistic Regression and Decision Tree
  - Plots decision tree diagram
  - Outputs classification report and confusion matrix



## 📊 Dataset Example (first 5 rows)

| id | age | job         | marital | education | default | balance | housing | loan |
|----|-----|-------------|---------|-----------|---------|---------|---------|------|
| 1  | 30  | unemployed  | married | primary   | no      | 1787    | no      | no   |
| 2  | 33  | services    | married | secondary | no      | 4789    | yes     | yes  |
| 3  | 35  | management  | single  | tertiary  | no      | 1350    | yes     | no   |
| 4  | 30  | management  | married | tertiary  | no      | 1476    | yes     | yes  |
| 5  | 59  | blue-collar | married | secondary | no      | 0       | yes     | no   |

- **Target column**: `y` (yes or no)
- **Columns**: 18 (age, job, balance, contact, etc.)
- **Missing values**: only in `age` (39 nulls)


## 🛠 How to Run

```bash
# Install required packages
pip install -r requirements.txt

# Run all classification models
python classifiers.py

# Run main demo (Logistic Regression & Decision Tree)
python main.py
````


📌 Conclusion
-------------------------------------------------------------------

- Best performing models (by accuracy): Logistic Regression, AdaBoost (0.90)
- Weakness: All models have low recall on positive class ("yes")



*This project offers a comprehensive benchmark for client subscription prediction, facilitating better marketing decisions with machine learning.*

---
