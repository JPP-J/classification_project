import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder, StandardScaler

def manual_hyperparameter_tuning(X, y):
    """
    Manual hyperparameter tuning using train-validation-test split
    """
    # Split data into train+val and test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Split remaining data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

    # Dictionary to store results
    results = []

    # Manual hyperparameter search
    n_estimators_list = [50, 100, 200]
    max_depth_list = [5, 10, None]

    for n_estimators in n_estimators_list:
        for max_depth in max_depth_list:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
            model.fit(X_train, y_train)

            val_pred = model.predict(X_val)
            val_score = r2_score(y_val, val_pred)

            results.append({
                'n_estimators': n_estimators,
                'max_depth': str(max_depth),
                'val_score': val_score
            })

    results_df = pd.DataFrame(results)
    print("Validation Results:")
    print(results_df)

    best_idx = results_df['val_score'].idxmax()
    best_n_estimators = int(results_df.loc[best_idx, 'n_estimators'])
    best_max_depth = results_df.loc[best_idx, 'max_depth']

    if best_max_depth == 'None':
        best_max_depth = None
    else:
        best_max_depth = int(best_max_depth)

    print("\nBest Parameters:")
    print(f"n_estimators: {best_n_estimators}")
    print(f"max_depth: {best_max_depth}")

    final_model = RandomForestRegressor(
        n_estimators=best_n_estimators,
        max_depth=best_max_depth,
        random_state=42
    )
    final_model.fit(X_train, y_train)

    test_pred = final_model.predict(X_test)
    test_score = r2_score(y_test, test_pred)
    print(f"\nFinal Test Score: {test_score:.4f}")

    return final_model, {'n_estimators': best_n_estimators, 'max_depth': best_max_depth}


def grid_search_tuning(X, y):
    """
    Hyperparameter tuning using GridSearchCV
    """
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': [1, 'sqrt', 'log2']  # Updated valid options
    }

    rf = RandomForestRegressor(random_state=42)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X, y)

    print("\nGrid Search Results:")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Score: {grid_search.best_score_:.4f}")

    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df[['params', 'mean_test_score', 'std_test_score']]
    print("\nAll Results:")
    print(results_df.sort_values('mean_test_score', ascending=False).head())

    return grid_search.best_estimator_, grid_search.best_params_


def random_search_tuning(X, y):
    """
    Hyperparameter tuning using RandomizedSearchCV
    """
    param_dist = {
        'n_estimators': np.arange(50, 300, 50),
        'max_depth': [5, 10, 15, None],
        'min_samples_split': np.arange(2, 11),
        'min_samples_leaf': np.arange(1, 6),
        'max_features': [1, 'sqrt', 'log2']  # Updated valid options
    }

    rf = RandomForestRegressor(random_state=42)

    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    random_search.fit(X, y)

    print("\nRandom Search Results:")
    print(f"Best Parameters: {random_search.best_params_}")
    print(f"Best Score: {random_search.best_score_:.4f}")

    return random_search.best_estimator_, random_search.best_params_


if __name__ == "__main__":
    # Load and prepare data
    # Set the option to display all columns
    pd.set_option('display.max_columns', None)

    np.random.seed(42)
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    print("1. Manual Hyperparameter Tuning:")
    manual_model, manual_params = manual_hyperparameter_tuning(X, y)

    print("\n2. Grid Search Tuning:")
    grid_model, grid_params = grid_search_tuning(X, y)

    print("\n3. Random Search Tuning:")
    random_model, random_params = random_search_tuning(X, y)

    # Compare final models
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nFinal Model Comparison (Test Set R² Scores):")
    models = {
        'Manual Tuning': manual_model,
        'Grid Search': grid_model,
        'Random Search': random_model
    }

    for name, model in models.items():
        test_pred = model.predict(X_test)
        score = r2_score(y_test, test_pred)
        print(f"{name}: {score:.4f}")