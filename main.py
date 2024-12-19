import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from utils.clsf_extended import plot_confusion_matrix, evaluate_clsf

# Part1: Load dataset
pd.set_option('display.max_columns', None)      # to show all columns - optional
path = "https://drive.google.com/uc?id=1QctGSSR5wSQk6cbdjBrKjYcUPs6PHNHN"
df = pd.read_csv(path)
print(f'Example of dataset:\n{df.head()}')
print(f'Columns name:\n{df.columns}')

# ------------------------------------------------------------------------------------------------------
# Part2: pre-processing
na = df.isnull().sum()      # missing value at ['age']
print(f'Null data:\n{na}')

# Fill null with mean values
df['age'] = df['age'].fillna(df['age'].mean())

# Focus on customer that age < 80
df = df[df['age'] < 80]

# Drop unnecessary attributes and assign to new df
df_n = df.drop(['id','contact','day','month','default','duration'], axis=1).reset_index(drop=True)

# Categorical features and numerical features
categorical_features = df_n.select_dtypes(include=[object]).columns.values
categorical_features = categorical_features[categorical_features != 'y']
numerical_features = df_n.select_dtypes(include=[np.number])

# Feature usages
features = df_n[['age', 'job', 'marital', 'education', 'balance', 'housing', 'loan',
                 'campaign', 'pdays', 'previous', 'poutcome']]

# Defines Parameters
X = features
y = df['y']

# Split data with test size 30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ------------------------------------------------------------------------------------------------------
# Part3: Train models and evaluation
# Define the ColumnTransformer for one-hot encoding
ct = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(drop='first', dtype=int), categorical_features)
    ],
    remainder='drop' # Dropping existing
)

# Model Decision Tree
# Define the Pipeline with the ColumnTransformer and the classifier
pipeline_dt = Pipeline(steps=[
    ('preprocessor', ct),
    ('classifier', tree.DecisionTreeClassifier(random_state=42, max_depth=3))
])

# Training
pipeline_dt.fit(X_train, y_train)
y_pred = pipeline_dt.predict(X_test)

# Evaluate
acc, report, cm = evaluate_clsf(y_test=y_test, y_pred=y_pred)
print('\nDecision Tree')
print(f'Accuracy Score: {acc:.4f} ')
print(f"\nClassification Report:\n{report}")
print(f"\nConfusion Matrix:\n{cm}")

# Get unique classes for consistent ordering
unique_classes = np.unique(y_train)
print(unique_classes)

# Extract the decision tree and feature names from the pipeline
decision_tree = pipeline_dt.named_steps['classifier']
feature_names = pipeline_dt.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out()

# VISUALIZATION
# Tree diagram
plt.figure(figsize=(12,8))
tree.plot_tree(decision_tree, feature_names=feature_names,
               class_names=unique_classes, filled=True, rounded=True)
plt.text(x=0.7,y=1,s= f"Accuracy score: {acc:.4f}%")
plt.show()

# Confusion Matrix
plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
plot_confusion_matrix(cm=cm, labels_cm=unique_classes, title='Decision Tree')

# ----------------------------------
# Model LogisticRegression
# Define the Pipeline with the ColumnTransformer and the classifier
pipeline_lg = Pipeline(steps=[
    ('preprocessor', ct),
    ('classifier',  LogisticRegression(solver='liblinear', max_iter=1000))
])

# Training
pipeline_lg.fit(X_train, y_train)
y_pred_lg = pipeline_lg.predict(X_test)

# Evaluate
acc_lg, report_lg, cm_lg = evaluate_clsf(y_test=y_test, y_pred=y_pred_lg)
print('\nLogisticRegression')
print(f'Accuracy Score: {acc_lg:.4f} ')
print(f"\nClassification Report:\n{report_lg}")
print(f"\nConfusion Matrix:\n{cm_lg}")

# VISUALIZATION
plt.subplot(1,2,2)
plot_confusion_matrix(cm=cm_lg, labels_cm=unique_classes, title='LogisticRegression')
plt.show()