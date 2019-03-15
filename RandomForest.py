# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 12:56:56 2019

@author: acostalago
"""

from sklearn.ensemble import RandomForestClassifier
from PreprocessData import X_train, X_test, y_train, y_test, labels
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=32)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print(cm)

feature_imp = pd.Series(clf.feature_importances_,index=labels).sort_values(ascending=False)
feature_imp

# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index, palette = 'PiYG_r', edgecolor = 'black')
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()