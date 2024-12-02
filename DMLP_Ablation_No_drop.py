import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split

# Load the training dataset
dataset = pd.read_csv("EbolaTrainblinddata.csv")
print(dataset)
X = dataset.iloc[:, 0:363].values  # Features
y = dataset.iloc[:, 363].values    # Target variable (labels)

# Load the testing dataset
dataset1 = pd.read_csv("blinddataset.csv")
print(dataset1)
Z = dataset1.iloc[:, 0:363].values  # Features for testing
k = dataset1.iloc[:, 363].values    # True labels for testing

# Define the Keras model with different variations for ablations

# Ablation 4: No dropout layers to prevent overfitting
model_4 = Sequential()
model_4.add(Dense(64, input_dim=363, activation='relu'))  # No dropout layers
model_4.add(Dense(32, activation='relu'))
model_4.add(Dense(16, activation='relu'))
model_4.add(Dense(8, activation='relu'))
model_4.add(Dense(1, activation='sigmoid'))  # No regularization techniques

# Compile and fit model_4 (no dropout layers)
model_4.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_4.fit(X, y, epochs=20, batch_size=32, verbose=1)

# Evaluate and make predictions for model_4 (no dropout layers)
_, accuracy_4 = model_4.evaluate(X, y)
print('Accuracy for model 4 (no dropout layers): %.2f' % (accuracy_4 * 100))
predict_x_4 = model_4.predict(Z)
yhat_classes_4 = (predict_x_4 > 0.5).astype(int)

# Compare the performance of model_4
accuracy = accuracy_score(k, yhat_classes_4)
precision = precision_score(k, yhat_classes_4)
recall = recall_score(k, yhat_classes_4)
f1 = f1_score(k, yhat_classes_4)
kappa = cohen_kappa_score(k, yhat_classes_4)
matrix = confusion_matrix(k, yhat_classes_4)

# ROC Curve and AUC for model_4
fpr, tpr, _ = roc_curve(k, predict_x_4)
roc_auc = auc(fpr, tpr)

# Precision-Recall Curve for model_4
precision_vals, recall_vals, _ = precision_recall_curve(k, predict_x_4)

# Plot the ROC Curve and Precision-Recall Curve for model_4
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

plt.subplot(1, 2, 2)
plt.plot(recall_vals, precision_vals, color='green', lw=2, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')

plt.tight_layout()
plt.show()

# Calculate AUC for Precision-Recall curve for model_4
pr_auc = auc(recall_vals, precision_vals)
print('PR AUC for model 4: %f' % pr_auc)
