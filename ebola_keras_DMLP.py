import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix, roc_curve, auc, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split

# Load the training dataset
dataset = pd.read_csv("EbolaTrainblinddata.csv")
#dataset = pd.read_csv("undersampled.csv")
#dataset = pd.read_csv("oversampled.csv")
print(dataset)
X = dataset.iloc[:, 0:363].values  # Features
y = dataset.iloc[:, 363].values    # Target variable (labels)

# Load the testing dataset
dataset1 = pd.read_csv("blinddataset.csv")
print(dataset1)
Z = dataset1.iloc[:, 0:363].values  # Features for testing
k = dataset1.iloc[:, 363].values    # True labels for testing

# Define the Keras model
model = Sequential()
model.add(Dense(64, input_dim=363, activation='relu'))  # Increased width of layers
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Binary classification output

# Compile the Keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model on the training dataset
model.fit(X, y, epochs=20, batch_size=32, verbose=1)  # Increased epochs and batch_size for better training

# Evaluate the model on the training data
_, accuracy = model.evaluate(X, y)
print('Accuracy for training: %.2f' % (accuracy * 100))

# Make class predictions with the model
predict_x = model.predict(Z)
yhat_classes = (predict_x > 0.5).astype(int)  # Threshold at 0.5 to get class labels (0 or 1)
print('Predicted classes:')
print(yhat_classes)

# Testing dataset true labels
print('Testing dataset true labels:')
print(k)

# Calculate evaluation metrics
accuracy = accuracy_score(k, yhat_classes)
print('Accuracy: %f' % accuracy)

precision = precision_score(k, yhat_classes)
print('Precision: %f' % precision)

recall = recall_score(k, yhat_classes)
print('Recall: %f' % recall)

f1 = f1_score(k, yhat_classes)
print('F1 score: %f' % f1)

kappa = cohen_kappa_score(k, yhat_classes)
print('Cohen\'s kappa: %f' % kappa)

matrix = confusion_matrix(k, yhat_classes)
print('Confusion Matrix:')
print(matrix)

# ROC Curve and AUC
fpr, tpr, _ = roc_curve(k, predict_x)
roc_auc = auc(fpr, tpr)
print('ROC AUC: %f' % roc_auc)

# Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(k, predict_x)

# Plot ROC Curve
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')

# Plot Precision-Recall Curve
plt.subplot(1, 2, 2)
plt.plot(recall_vals, precision_vals, color='green', lw=2, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')

plt.tight_layout()
plt.show()

# Calculate AUC for Precision-Recall curve
pr_auc = auc(recall_vals, precision_vals)
print('PR AUC: %f' % pr_auc)
