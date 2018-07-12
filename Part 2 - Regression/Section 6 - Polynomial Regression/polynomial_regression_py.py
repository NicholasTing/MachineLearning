#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 13:11:46 2018

@author: nicholas
"""

# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
# Sometimes, it wouldnt make much sense to split train and test set
# Especially when there is little data
"""
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Linear Regression to dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to dataset
from sklearn.preprocessing import PolynomialFeatures
# degree = 2
poly_reg = PolynomialFeatures(degree = 2)

# First fit, then transform
# New matrix X poly.
X_poly = poly_reg.fit_transform(X)

# Create a second linear regression
line_reg_2 = LinearRegression()
line_reg_2.fit(X_poly, y)

# Visualising the Linear Regression results

# Visualising the Polynomial Regression results