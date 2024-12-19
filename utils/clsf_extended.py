import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def plot_confusion_matrix(cm, labels_cm, title=''):
    # plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',cbar=False,
                xticklabels=labels_cm,
                yticklabels=labels_cm)
    plt.xlabel('Predicted Label', color='red')
    plt.ylabel('True Label', color='blue')
    plt.title(f'Confusion Matrix {title}')


def evaluate_clsf(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred)
    cm_matrix = confusion_matrix(y_test, y_pred)
    return acc , clf_report, cm_matrix

