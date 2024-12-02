import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
dataset = pd.read_csv("EbolaTrainblinddata.csv")
print(dataset.head())  # Print the first few rows to understand the dataset

X = dataset.iloc[:, 0:363].values
Y = dataset.iloc[:, 363].values

dataset1 = pd.read_csv("blinddataset.csv")
Z = dataset1.iloc[:, 0:363].values
k = dataset1.iloc[:, 363].values

# Define and train the AdaBoost model
base_estimator = DecisionTreeClassifier(max_depth=1)  # Weak learner
adaboost = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=50, learning_rate=1.0)
adaboost.fit(X, Y)

# Make predictions on validation dataset
predictions = adaboost.predict(Z)

# Evaluate the model
accuracy = accuracy_score(k, predictions)
print(f'Accuracy score of AdaBoost: {accuracy}')
print('Confusion Matrix:')
print(confusion_matrix(k, predictions))
print('Classification Report:')
print(classification_report(k, predictions))

# Calculate ROC AUC and Precision-Recall AUC (for binary classification)
if len(np.unique(k)) == 2:  # Binary classification check
    probas_ = adaboost.predict_proba(Z)[:, 1]
    fpr, tpr, _ = roc_curve(k, probas_)
    roc_auc = roc_auc_score(k, probas_)
    print(f'ROC AUC score for AdaBoost: {roc_auc}')

    # Plot ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - AdaBoost')
    plt.legend(loc="lower right")
    plt.show()

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(k, probas_)
    pr_auc = roc_auc_score(k, probas_)  # PR-AUC can often use ROC-AUC for binary
    print(f'PR AUC score for AdaBoost: {pr_auc}')

    # Plot Precision-Recall Curve
    plt.figure()
    plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - AdaBoost')
    plt.legend(loc="lower left")
    plt.show()
