#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 21:51:14 2018

@author: nicholas
"""

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Data.csv')

#on the left, we take all the lines. all the rows
#on the right, take all the columns except the last one
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values