import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

# Part1: Load dataset
pd.set_option('display.max_columns', None)      # to show all columns - optional
path = "https://drive.google.com/uc?id=1QctGSSR5wSQk6cbdjBrKjYcUPs6PHNHN"
df = pd.read_csv(path)
print(f'Example of dataset:\n{df.head()}')
print(f'Columns name:\n{df.columns}')

# -------------------------------------------------------------------------------------------------------
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
X_encoded = pd.get_dummies(X, columns=categorical_features)
y = df['y']

# Split data with test size 30%
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

# -------------------------------------------------------------------------------------------------------
# Part3: Train models and evaluation
# Create a list of classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(solver='liblinear', max_iter=1000), # default: lbfgs, other: 'liblinear' or 'saga'
    "Decision Tree": DecisionTreeClassifier(max_depth=5),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(),
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier()
}

# Train and evaluate each classifier
for name, clf in classifiers.items():
    # Train the model
    # model = make_pipeline(StandardScaler(), clf)      # in case need standardize data - optional
    model = clf
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name} Accuracy: {accuracy:.2f}')

    # Confusion Matrix
    print(f'{name} Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    print('-' * 40)