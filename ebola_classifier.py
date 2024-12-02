import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from numpy import loadtxt

# load the dataset
dataset=pd.read_csv("oversampled.csv");
print(dataset)
X = dataset.iloc[:,0:363].values
Y = dataset.iloc[:,363].values

dataset1=pd.read_csv("blinddataset.csv");
Z = dataset1.iloc[:,0:363].values
k = dataset1.iloc[:,363].values

#validation_size = 0.20
#seed = 7
#X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, #test_size=validation_size)

scoring = 'accuracy'
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('NB', GaussianNB()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier(n_estimators=100)))
models.append(('SVM', SVC()))
models.append(('ADB',AdaBoostClassifier(n_estimators=50,learning_rate=1)))
models.append(('XGB',XGBClassifier()))
models.append(('LDR',LinearDiscriminantAnalysis()))
models.append(('LR',LogisticRegression()))

'''# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10)
	cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print("The accuracy is:",msg)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()'''


# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X,Y)
predictions = knn.predict(Z)
print('Accuracy score of KNN:',accuracy_score(k, predictions))
print(confusion_matrix(k, predictions))
print('classification report:',classification_report(k, predictions))

LR = LogisticRegression()
LR.fit(X, Y)
predictions = LR.predict(Z)
print('Accuracy score of LR:',accuracy_score(k, predictions))
print(confusion_matrix(k, predictions))
print('classification report:',classification_report(k, predictions))

gb = GaussianNB()
gb.fit(X, Y)
predictions = gb.predict(Z)
print('Accuracy score of NB:',accuracy_score(k, predictions))
print(confusion_matrix(k, predictions))
print('classification report:',classification_report(k, predictions))

dct = DecisionTreeClassifier()
dct.fit(X, Y)
predictions = dct.predict(Z)
print('Accuracy score of DCT:',accuracy_score(k, predictions))
print(confusion_matrix(k, predictions))
print('classification report:',classification_report(k, predictions))

ldr = LinearDiscriminantAnalysis()
ldr.fit(X, Y)
predictions = ldr.predict(Z)
print('Accuracy score of LDR:',accuracy_score(k, predictions))
print(confusion_matrix(k, predictions))
print('classification report:',classification_report(k, predictions))

svm = SVC()
svm.fit(X, Y)
predictions = svm.predict(Z)
print('Accuracy score of SVM:',accuracy_score(k, predictions))
print(confusion_matrix(k, predictions))
print('classification report:',classification_report(k, predictions))

# Make predictions on validation dataset
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, Y)
predictions = rf.predict(Z)
print('Accuracy score of RandomForest:',accuracy_score(k, predictions))
print(confusion_matrix(k, predictions))
print('classification report:',classification_report(k, predictions))


# Make predictions on validation dataset
xgb = XGBClassifier()
xgb.fit(X, Y)
predictions = xgb.predict(Z)
print('Accuracy score of XGBoost:',accuracy_score(k, predictions))
print(confusion_matrix(k, predictions))
print('classification report:',classification_report(k, predictions))

# Make predictions on validation dataset
ada = AdaBoostClassifier(n_estimators=50,learning_rate=1)
ada.fit(X, Y)
predictions = ada.predict(Z)
print('Accuracy score of AdaBoost:',accuracy_score(k, predictions))
print(confusion_matrix(k, predictions))
print('classification report:',classification_report(k, predictions))