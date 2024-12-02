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

# Ablation 1: Removing hidden layers (only input and output layer)
model_1 = Sequential()
model_1.add(Dense(1, input_dim=363, activation='sigmoid'))  # No hidden layers

# Ablation 2: Using only one hidden layer (i.e., fewer layers than the original model)
model_2 = Sequential()
model_2.add(Dense(64, input_dim=363, activation='relu'))  # Single hidden layer
model_2.add(Dense(1, activation='sigmoid'))  # Binary classification output

# Ablation 3: Using a different activation function (tanh instead of ReLU)
model_3 = Sequential()
model_3.add(Dense(64, input_dim=363, activation='tanh'))  # tanh activation function
model_3.add(Dense(1, activation='sigmoid'))

# Ablation 4: No dropout layers to prevent overfitting (this is an optional modification)
# model_4 = Sequential()
# model_4.add(Dense(64, input_dim=363, activation='relu'))  # Dropout not included
# model_4.add(Dense(32, activation='relu'))
# model_4.add(Dense(16, activation='relu'))
# model_4.add(Dense(8, activation='relu'))
# model_4.add(Dense(1, activation='sigmoid'))  # No regularization techniques

# Compile and fit model_1 (no hidden layers)
model_1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_1.fit(X, y, epochs=20, batch_size=32, verbose=1)

# Evaluate and make predictions for model_1
_, accuracy_1 = model_1.evaluate(X, y)
print('Accuracy for model 1 (no hidden layers): %.2f' % (accuracy_1 * 100))
predict_x_1 = model_1.predict(Z)
yhat_classes_1 = (predict_x_1 > 0.5).astype(int)

# Compile and fit model_2 (single hidden layer)
model_2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_2.fit(X, y, epochs=20, batch_size=32, verbose=1)

# Evaluate and make predictions for model_2
_, accuracy_2 = model_2.evaluate(X, y)
print('Accuracy for model 2 (single hidden layer): %.2f' % (accuracy_2 * 100))
predict_x_2 = model_2.predict(Z)
yhat_classes_2 = (predict_x_2 > 0.5).astype(int)

# Compile and fit model_3 (with tanh activation function)
model_3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_3.fit(X, y, epochs=20, batch_size=32, verbose=1)

# Evaluate and make predictions for model_3
_, accuracy_3 = model_3.evaluate(X, y)
print('Accuracy for model 3 (tanh activation): %.2f' % (accuracy_3 * 100))
predict_x_3 = model_3.predict(Z)
yhat_classes_3 = (predict_x_3 > 0.5).astype(int)

# Compare the performance of the models (example for model_1)
accuracy = accuracy_score(k, yhat_classes_1)
precision = precision_score(k, yhat_classes_1)
recall = recall_score(k, yhat_classes_1)
f1 = f1_score(k, yhat_classes_1)
kappa = cohen_kappa_score(k, yhat_classes_1)
matrix = confusion_matrix(k, yhat_classes_1)

# ROC Curve and AUC for model_1
fpr, tpr, _ = roc_curve(k, predict_x_1)
roc_auc = auc(fpr, tpr)

# Precision-Recall Curve for model_1
precision_vals, recall_vals, _ = precision_recall_curve(k, predict_x_1)

# Plot the ROC Curve and Precision-Recall Curve for model_1
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

# Calculate AUC for Precision-Recall curve for model_1
pr_auc = auc(recall_vals, precision_vals)
print('PR AUC for model 1: %f' % pr_auc)

