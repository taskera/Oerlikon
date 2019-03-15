# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 18:47:38 2019

@author: acostalago
"""
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm
from PreprocessData import X_train, y_train, X_test, y_test

""" Creating the logistic regression model"""
# Statistical model of the logistic regression to provide table with the results
logit_model=sm.Logit(y_train, X_train)
result=logit_model.fit()
print(result.summary2())

# Logistic regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Prediction with the test data splitted previously
predictions = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

# Confusion matrix
cm = confusion_matrix(y_test, predictions)
print(cm)

# Receiver operating characteristic curve (ROC curve)
logit_roc_auc = roc_auc_score(y_test, predictions)
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('Receiver operating characteristic', fontsize=18)
plt.legend(loc="lower right")