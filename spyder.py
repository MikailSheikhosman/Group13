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
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, silhouette_score

plt.style.use('seaborn-v0_8-whitegrid')

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.show()

def plot_complexity_curve(estimator, title, X, y, param_name, param_range, cv=3, scoring="f1"):
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring=scoring, n_jobs=1
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(8, 5))
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel(f"Score ({scoring})")
    plt.ylim(0.0, 1.1)
    
    plt.plot(param_range, train_mean, label="Training Score", color="darkorange", marker='o')
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.2, color="darkorange")
    
    plt.plot(param_range, test_mean, label="Cross-Validation Score", color="navy", marker='o')
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.2, color="navy")
    
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()

# Load with semicolon delimiter as seen in dataset
df = pd.read_csv('bank-full.csv', sep=';')


if 'duration' in df.columns:
    df = df.drop('duration', axis=1)






# Calculate probabilities of classes
counts = df['y'].value_counts(normalize=True)
prob_no = counts['no']
prob_yes = counts['yes']

#  Majority Class (Always predict 'no')
baseline_majority = prob_no

#  Random Classifier (p(yes)^2 + p(no)^2)
baseline_random = (prob_yes ** 2) + (prob_no ** 2)

print(f"Class Distribution: No= {prob_no:.4f}, Yes= {prob_yes:.4f}")
print(f"Majority Class:   {baseline_majority*100:.2f}%")
print(f"Random Classifier: {baseline_random*100:.2f}% ")






X_raw = df.drop('y', axis=1)
y_raw = df['y']


le = LabelEncoder()
y = le.fit_transform(y_raw)

X_encoded = pd.get_dummies(X_raw, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.3, stratify=y, random_state=42)




scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)





# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='coolwarm', alpha=0.6, s=10)
plt.colorbar(scatter, label='Term Deposit (0=No, 1=Yes)')
plt.title('PCA Projection (2 Components)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

results = {}

# Silhouette Scores to find optimal k
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans_temp.fit_predict(X_train_scaled)
    score = silhouette_score(X_train_scaled, cluster_labels)
    silhouette_scores.append(score)

# Plot Silhouette Scores
plt.figure(figsize=(8, 5))
plt.plot(k_range, silhouette_scores, marker='o', linestyle='-', color='green')
plt.title('Silhouette Scores for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score (Higher is Better)')
plt.grid(True)
plt.show()

best_index = np.argmax(silhouette_scores)
optimal_k = k_range[best_index]

print(f"Optimal k based on Silhouette Score: {optimal_k}")

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_final.fit(X_train_scaled)

train_cluster_ids = kmeans_final.predict(X_train_scaled)
cluster_map = {}

for cluster_id in range(optimal_k):
    indices = np.where(train_cluster_ids == cluster_id)
    true_labels_in_cluster = y_train[indices]
    
    if len(true_labels_in_cluster) > 0:
        mode_label = np.bincount(true_labels_in_cluster).argmax()
    else:
        mode_label = 0
    
    cluster_map[cluster_id] = mode_label

test_cluster_ids = kmeans_final.predict(X_test_scaled)
y_pred_kmeans = np.array([cluster_map[cid] for cid in test_cluster_ids])
results[f'KMeans (k={optimal_k})'] = y_pred_kmeans

# Decision Tree 
dt = DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=7)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
results['Decision Tree'] = y_pred_dt

# Decision Tree Feature Importance Plot
importances_dt = dt.feature_importances_
indices_dt = np.argsort(importances_dt)[::-1][:10]
plt.figure(figsize=(10, 5))
plt.title("Top 10 Feature Importances (Decision Tree)")
plt.bar(range(10), importances_dt[indices_dt], align="center", color='salmon')
plt.xticks(range(10), X_encoded.columns[indices_dt], rotation=45)
plt.tight_layout()
plt.show()





# Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', max_depth=15)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
results['Random Forest'] = y_pred_rf


importances = rf.feature_importances_
indices = np.argsort(importances)[::-1][:10]

# Plot Feature Importance
plt.figure(figsize=(10, 5))
plt.title("Top 10 Feature Importances (Random Forest)")
plt.bar(range(10), importances[indices], align="center", color='skyblue')
plt.xticks(range(10), X_encoded.columns[indices], rotation=45)
plt.tight_layout()
plt.show()


# lean forest
top_features = X_encoded.columns[indices]
print(f"     Selected Features: {list(top_features)}")

X_train_lean = X_train[top_features]
X_test_lean = X_test[top_features]
rf_lean = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', max_depth=15)
rf_lean.fit(X_train_lean, y_train)
results['RF (Top 10 Features)'] = rf_lean.predict(X_test_lean)


#tuned forest
rf_params = {
    'n_estimators': [50, 100, 150, 200, 250, 300], 
    'max_depth': [5, 10, 15, 20, 25],
    'min_samples_split': [2, 5],     
    'max_features': ['sqrt', 'log2'] 
}

# Run GridSearch
rf_grid = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42), 
                       rf_params, cv=3, scoring='f1', n_jobs=1)
rf_grid.fit(X_train[:5000], y_train[:5000])

print(f"     Best RF Params: {rf_grid.best_params_}")

best_rf = rf_grid.best_estimator_
best_rf.fit(X_train, y_train)
results['RF (Tuned)'] = best_rf.predict(X_test)





# Naive bayes
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
y_pred_nb = nb.predict(X_test_scaled)
results['Naive Bayes'] = y_pred_nb





# Perceptron 
ppn = Perceptron(max_iter=1000, eta0=0.1, random_state=42)
ppn.fit(X_train_scaled, y_train)
y_pred_ppn = ppn.predict(X_test_scaled)
results['Perceptron'] = y_pred_ppn



# k-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
results['k-NN'] = y_pred_knn







# SVM
#optimisation
param_grid = {
    'C': [0.1, 1, 3, 5, 8, 10], 
    'gamma': ['scale', 'auto'], 
    'kernel': ['rbf', 'poly']
}


# n_jobs=1 prevents the ChildProcessError
grid = GridSearchCV(SVC(class_weight='balanced', random_state=42), 
                    param_grid, refit=True, verbose=0, cv=3, n_jobs=1)


grid.fit(X_train_scaled[:5000], y_train[:5000])

print(f"     Best SVM Params: {grid.best_params_}")


best_svm = grid.best_estimator_
best_svm.fit(X_train_scaled, y_train)


y_pred_svm = best_svm.predict(X_test_scaled)
results['SVM'] = y_pred_svm


# NN 

# SMOTE used

sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train_scaled, y_train)


mlp_params = {
    'hidden_layer_sizes': [(50, 50), (100,), (100, 50)], 
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],  
    'learning_rate_init': [0.001, 0.01]
}


mlp_grid = GridSearchCV(
    MLPClassifier(max_iter=1000, random_state=42, early_stopping=True),
    mlp_params, 
    cv=3, 
    scoring='f1', 
    n_jobs=-1
)

mlp_grid.fit(X_train_resampled, y_train_resampled)


best_mlp = mlp_grid.best_estimator_
y_pred_mlp_opt = best_mlp.predict(X_test_scaled)
results['Neural Network '] = y_pred_mlp_opt


plt.figure(figsize=(6, 4))
plt.plot(best_mlp.loss_curve_)
plt.title("Optimized Neural Network Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()





print("RESULTS TABLE")


print(f"{'Model':<20} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
print("-" * 75)

final_f1_scores = []
final_model_names = []

for name, y_pred in results.items():
    # Calculate all metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0) 
    rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)     
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)    
          
    if 'KMeans' not in name:
        final_f1_scores.append(f1)
        final_model_names.append(name)
   
    print(f"{name:<20} | {acc:.4f}     | {prec:.4f}     | {rec:.4f}     | {f1:.4f}")
    
    # Plot Confusion Matrix
    plot_confusion_matrix(y_test, y_pred, title=f"Confusion Matrix: {name}")


plt.figure(figsize=(12, 6))
sns.barplot(x=final_f1_scores, y=final_model_names, palette='magma')
plt.title('Model Performance Comparison (F1-Score)')
plt.xlabel('F1 Score')
plt.xlim(0, 1.0)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()








#gi









































































