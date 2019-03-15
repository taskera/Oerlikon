# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 19:02:16 2019

@author: acostalago
"""

from LoadData import dataset
from sklearn.preprocessing import LabelEncoder


""" Encoding categorical data """
# Encoding of the dependant variable 0 -> functional, 1-> non functional
y = dataset['status_group']
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

""" Columns of interest: amount_tsh, installer, basin, population, 
public_meeting, scheme_management, permit, Year, Type_class, 
management_group, payment_type, quality_group, quantity_group, source_type, 
source_class, waterpoint_type_group """
labels = ['amount_tsh', 'basin', 'population', \
             'public_meeting', 'scheme_management', 'permit', 'Year', \
             'Type_class', 'management_group', 'payment_type', 'quality_group',\
             'quantity_group', 'source_type', 'source_class', \
             'waterpoint_type_group', 'installer']
X = dataset[labels].values

# Encoding of the inputs
# Encoding of (column 2) Basin (9 values)
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

# Encoding of (column 4) public_meeting (0 -> true, 1 -> False)
labelencoder_X_2 = LabelEncoder()
X[:, 3] = labelencoder_X_2.fit_transform(X[:, 3])

# Encoding of (column 5) scheme_management (13 values)
labelencoder_X_3 = LabelEncoder()
X[:, 4] = labelencoder_X_3.fit_transform(X[:, 4])

# Encoding of (column 6) permit (0 -> true, 1 -> False)
labelencoder_X_4 = LabelEncoder()
X[:, 5] = labelencoder_X_4.fit_transform(X[:, 5])

# Encoding of (column 8) Type_class (7 values)
labelencoder_X_5 = LabelEncoder()
X[:, 7] = labelencoder_X_5.fit_transform(X[:, 7])

# Encoding of (column 9) management_group (5 values)
labelencoder_X_6 = LabelEncoder()
X[:, 8] = labelencoder_X_6.fit_transform(X[:, 8])

# Encoding of (column 10) payment_type (7 values)
labelencoder_X_7 = LabelEncoder()
X[:, 9] = labelencoder_X_7.fit_transform(X[:, 9])

# Encoding of (column 11) quality_group (6 values)
labelencoder_X_8 = LabelEncoder()
X[:, 10] = labelencoder_X_8.fit_transform(X[:, 10])

# Encoding of (column 12) quantity_group (5 values)
labelencoder_X_9 = LabelEncoder()
X[:, 11] = labelencoder_X_9.fit_transform(X[:, 11])

# Encoding of (column 13) source_type (7 values)
labelencoder_X_10 = LabelEncoder()
X[:, 12] = labelencoder_X_10.fit_transform(X[:, 12])

# Encoding of (column 14) source_class (3 values)
labelencoder_X_11 = LabelEncoder()
X[:, 13] = labelencoder_X_11.fit_transform(X[:, 13])

# Encoding of (column 15) waterpoint_type_group (6 values)
labelencoder_X_12 = LabelEncoder()
X[:, 14] = labelencoder_X_12.fit_transform(X[:, 14])

# Encoding of (column 16) installer (~200 values)
labelencoder_X_13 = LabelEncoder()
X[:, 15] = labelencoder_X_13.fit_transform(X[:, 15])

# Feature Scaling
# Remove the mean and scale to unit variance
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
