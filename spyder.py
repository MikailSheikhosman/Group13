#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 22:38:42 2025

@author: apps
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

plt.style.use('seaborn-v0_8-whitegrid')

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """
    Helper function to plot a confusion matrix.
    Recommended by marker feedback.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.show()


# Load with semicolon delimiter as seen in dataset
df = pd.read_csv('bank-full.csv', sep=';')

# Drop 'duration' column (Data Leakage) The duration is not known before a call is performed.
if 'duration' in df.columns:
    df = df.drop('duration', axis=1)




