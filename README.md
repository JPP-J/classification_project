# Classification Project
![Last Commit](https://img.shields.io/github/last-commit/JPP-J/classification_project?style=flat-square)
![Python](https://img.shields.io/badge/Python-96.2%25-blue?style=flat-square)
![Languages](https://img.shields.io/github/languages/count/JPP-J/classification_project?style=flat-square)

This repo is home to the code that accompanies Jidapa's *Classification Project*. Compares multiple machine learning classification models on a [**bank marketing dataset**](https://drive.google.com/file/d/1QctGSSR5wSQk6cbdjBrKjYcUPs6PHNHN/view?usp=drive_link) with [detail of dataset](https://drive.google.com/file/d/1K2wneqZNolblPX2WBbC2keoxZSg7ATYe/view?usp=sharing)) to identify the best-performing algorithm. The dataset includes client information and marketing campaign outcomes, aiming to predict whether a client will subscribe to a term deposit.

---

## 📂 Project Files

### [`classifiers.py`](classifiers.py)
Trains and evaluates the following classification models:

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors
- Support Vector Machine
- Random Forest
- Naive Bayes
- Gradient Boosting
- AdaBoost

Each model reports its accuracy and confusion matrix.

### [`main.py`](main.py)
- Loads and preprocesses the bank dataset
- Displays a preview and dataset statistics
- Trains and evaluates Logistic Regression and Decision Tree
- Plots decision tree diagram
- Outputs classification report and confusion matrix

---

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

---

## ✅ Model Performance Summary [`classifiers.py`](classifiers.py)

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

---

## 📌 Detailed Results from [`main.py`](main.py)

### Logistic Regression

- **Accuracy**: 0.90 
- **Confusion Matrix**:

- **Classification Report**: Logistic Regression
  - Accuracy: 0.90
  - Confusion Matrix:
      [[1186   12]
       [ 129   26]]
  - Precision (yes): 0.68
  - Recall (yes): 0.17
  - F1-score (yes): 0.27
    
## 🛠 How to Run

```bash
# Install required packages
pip install -r requirements.txt

# Run all classification models
python classifiers.py

# Run main demo (Logistic Regression & Decision Tree)
python main.py
````

-------------------------------------------------------------------
📌 Conclusion
-------------------------------------------------------------------

- Best performing models (by accuracy): Logistic Regression, AdaBoost (0.90)
- Weakness: All models have low recall on positive class ("yes")
"""
