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
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, silhouette_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import gc

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


y_for_interpret = df["y"]
X_for_clustering = df.drop(columns=["y"])


#KMEANS PCA
# turned categorical columns into numeric features 
#  preprocessing so can apply to the dataset.
X_numeric = pd.get_dummies(X_for_clustering, drop_first=True)

# scaled features 
scaler_k = StandardScaler()
X_scaled = scaler_k.fit_transform(X_numeric)


# pca so representation , visualisation , decide components using variance explained
pca_full = PCA(random_state=42)
X_pca_full = pca_full.fit_transform(X_scaled)

# plott cumulative proportion of variance explained
cum_pve = np.cumsum(pca_full.explained_variance_ratio_)
plt.figure(figsize=(7, 4))
plt.plot(range(1, len(cum_pve) + 1), cum_pve, marker="o", markersize=3)
plt.xlabel("Number of PCA components")
plt.ylabel("Cumulative proportion of variance explained")
plt.title("PCA: cumulative variance explained ")
plt.tight_layout()
plt.show()

# Picked a number of components for clustering 
# tweked was 15 ,25 now 30
pca_dims_for_kmeans = 30
pca_for_kmeans = PCA(n_components=pca_dims_for_kmeans, random_state=42)
X_pca_for_kmeans = pca_for_kmeans.fit_transform(X_scaled)

# 2d pca for plotting clusters
pca_2d = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], s=8, alpha=0.35)
plt.title("PCA (2D) projection before clustering")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()


# K-MEANS 
k_list = [2, 3, 4, 5, 6] 
chosen_k = 4

print("\n K MEANS  RESULTS ")
print(f"PCA dimensions used for k means: {pca_dims_for_kmeans}")
print("Trying different k values (…\n")

for k in k_list:
    km = KMeans(n_clusters=k, random_state=42, n_init=50, max_iter=500)
    cluster_ids = km.fit_predict(X_pca_for_kmeans)

    centroids = km.cluster_centers_
    sse = float(np.sum((X_pca_for_kmeans - centroids[cluster_ids]) ** 2))

    # Show clusters on PCA 2D projection 
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=cluster_ids, s=8, alpha=0.5)
    plt.title(f"K-means clusters on PCA(2D) plot (k={k})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(label="Cluster")
    plt.tight_layout()
    plt.show()

    # Interpretation only,  y is used after clustering
    cluster_summary = (
        pd.DataFrame({"cluster": cluster_ids, "y": y_for_interpret})
        .groupby("cluster")
        .agg(
            n=("cluster", "size"),
            yes_rate=("y", lambda s: (s == "yes").mean())
        )
        .sort_values("yes_rate", ascending=False)
    )

    print(f"k = {k}")
    print(f"Within cluster SSE : {sse:.2f}")
    print("Cluster sizes + yes_rate (interpretation only):")
    print(cluster_summary)
   
    
    if k == chosen_k:
        #  cluster profiling for the chosen solution (k=4) 
        df_profile = df.copy()
        df_profile["cluster"] = cluster_ids

        # Numeric averages by cluster
        num_cols = df_profile.select_dtypes(include=[np.number]).columns
        
        print(f"\n numeric means by cluster (k={chosen_k}):")
        
        print(df_profile.groupby("cluster")[num_cols].mean(numeric_only=True))

        # Most common categories by cluster (mode)
        cat_cols = df_profile.select_dtypes(exclude=[np.number]).columns
        cat_cols = [c for c in cat_cols if c not in ["y"]]
        print(f"\n most common categories by cluster (k={chosen_k}):")
        print(df_profile.groupby("cluster")[cat_cols].agg(lambda s: s.mode().iloc[0]))

        # Yes rate by cluster
        print(f"\nyes rate by cluster (k={chosen_k}):")

        print(df_profile.groupby("cluster")["y"]
              .apply(lambda s: (s == "yes").mean())
              .sort_values(ascending=False))

num_cols = [c for c in df_profile.select_dtypes(include=[np.number]).columns if c != "cluster"]
print(df_profile.groupby("cluster")[num_cols].mean(numeric_only=True))


# Stability


print("stability check")
print(f"Chosen k for stability check: {chosen_k}")

stability_rows = []
for seed in [1, 2, 3, 4, 5]:
    km_s = KMeans(n_clusters=chosen_k, random_state=seed, n_init=50, max_iter=500)

    labels_s = km_s.fit_predict(X_pca_for_kmeans)
    cents_s = km_s.cluster_centers_
    sse_s = float(np.sum((X_pca_for_kmeans - cents_s[labels_s]) ** 2))

    sizes = pd.Series(labels_s).value_counts().sort_index().to_dict()
    stability_rows.append({"seed": seed, "within_sse": sse_s, "cluster_sizes": sizes})

stability_df = pd.DataFrame(stability_rows)
print(stability_df.to_string(index=False))





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

# Logistic Regression
log_reg = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)
results['Logistic Regression'] = y_pred_lr


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

#neural network
np.random.seed(42)
tf.random.set_seed(42)

#  Train/Validation split
X_train_nn, X_val_nn, y_train_nn, y_val_nn = train_test_split(
    X_train_scaled, y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

# One-hot targets for softmax ,categorical_crossentropy 
y_train_oh = keras.utils.to_categorical(y_train_nn, num_classes=2)
y_val_oh   = keras.utils.to_categorical(y_val_nn,   num_classes=2)

def build_mlp(input_dim, units=(128, 64), activation="relu",
              dropout_rate=0.2, l2_strength=0.0):
    reg = regularizers.l2(l2_strength) if l2_strength > 0 else None

    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    for u in units:
        model.add(layers.Dense(u, activation=activation, kernel_regularizer=reg))
        if dropout_rate and dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(2, activation="softmax"))
    return model

def make_optimizer(opt_name, lr, momentum=0.9):
    if opt_name == "adam":
        return keras.optimizers.Adam(learning_rate=lr)
    elif opt_name == "sgd":
        return keras.optimizers.SGD(learning_rate=lr, momentum=momentum)
    else:
        raise ValueError("opt_name must be 'adam' or 'sgd'")

def make_early_stop(patience=8, min_delta=1e-3):
    return keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=min_delta,
        patience=patience,
        restore_best_weights=True
    )

def best_threshold_for_f1(y_true, p_yes):
    # threshold search to maximise F1 on validation
    best_t, best_f1 = 0.5, -1
    for t in np.linspace(0.05, 0.95, 19): 
        y_pred = (p_yes >= t).astype(int)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1

# 3) Small search space 
search_space = [
    # Architecture/regularisation knobs (dropout/L2) + optimiser knobs (Adam/SGD+momentum)
    {"units": (128, 64), "activation": "relu", "dropout": 0.1, "l2": 0.0,   "opt": "adam", "lr": 1e-3,  "batch": 128},
    {"units": (128, 64), "activation": "relu", "dropout": 0.2, "l2": 0.0,   "opt": "adam", "lr": 5e-4,  "batch": 128},
    {"units": (128, 64), "activation": "relu", "dropout": 0.2, "l2": 1e-4,  "opt": "adam", "lr": 5e-4,  "batch": 128},

    {"units": (128, 64), "activation": "tanh", "dropout": 0.1, "l2": 0.0,   "opt": "adam", "lr": 5e-4,  "batch": 128},
    {"units": (128, 64), "activation": "tanh", "dropout": 0.2, "l2": 1e-4,  "opt": "adam", "lr": 5e-4,  "batch": 128},

    {"units": (128, 64), "activation": "relu", "dropout": 0.2, "l2": 0.0,   "opt": "sgd",  "lr": 1e-2,  "momentum": 0.9, "batch": 128},
    {"units": (128, 64), "activation": "relu", "dropout": 0.1, "l2": 0.0,   "opt": "sgd",  "lr": 1e-2,  "momentum": 0.9, "batch": 128},
    {"units": (128, 64), "activation": "relu", "dropout": 0.2, "l2": 1e-4,  "opt": "sgd",  "lr": 1e-2,  "momentum": 0.9, "batch": 128},

    # slightly smaller model. sometimes generalises better
    {"units": (64, 64),  "activation": "relu", "dropout": 0.1, "l2": 0.0,   "opt": "adam", "lr": 5e-4,  "batch": 128},
    {"units": (64, 64),  "activation": "relu", "dropout": 0.2, "l2": 1e-4,  "opt": "sgd",  "lr": 1e-2,  "momentum": 0.9, "batch": 128},
]

early_stop = make_early_stop(patience=8, min_delta=1e-3)

print("\nNEURAL NETWORK (lecture-style tuning + threshold tuning)")
print(f"Total configs to try: {len(search_space)}\n")

best = {"val_f1": -1, "cfg": None, "model": None, "threshold": 0.5}

for i, cfg in enumerate(search_space, start=1):
    keras.backend.clear_session()
    gc.collect()

    model = build_mlp(
        input_dim=X_train_nn.shape[1],
        units=cfg["units"],
        activation=cfg["activation"],
        dropout_rate=cfg["dropout"],
        l2_strength=cfg["l2"]
    )

    opt = make_optimizer(
        opt_name=cfg["opt"],
        lr=cfg["lr"],
        momentum=cfg.get("momentum", 0.9)
    )

    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(
        X_train_nn, y_train_oh,
        validation_data=(X_val_nn, y_val_oh),
        epochs=120,                # early stopping
        batch_size=cfg["batch"],
        verbose=0,
        callbacks=[early_stop]
    )

    # validation . tune threshold for best F1 
    val_probs = model.predict(X_val_nn, verbose=0)
    p_yes = val_probs[:, 1]
    t_best, f1_best = best_threshold_for_f1(y_val_nn, p_yes)

    # also report the default argmax metrics 
    val_pred_default = np.argmax(val_probs, axis=1)
    val_f1_default = f1_score(y_val_nn, val_pred_default, pos_label=1, zero_division=0)

    print(f"[{i:02d}/{len(search_space)}] {cfg} -> "
          f"Val F1 (default argmax)={val_f1_default:.4f} | "
          f"Val F1 (best threshold {t_best:.2f})={f1_best:.4f}")

    if f1_best > best["val_f1"]:
        best.update({"val_f1": f1_best, "cfg": cfg, "model": model, "threshold": t_best})

print("\nBEST CONFIG (by validation F1 with tuned threshold):")
print(best["cfg"])
print(f"Best validation F1: {best['val_f1']:.4f} at threshold={best['threshold']:.2f}")

#  Final test evaluation using tuned threshold 
best_model = best["model"]
test_probs = best_model.predict(X_test_scaled, verbose=0)
p_yes_test = test_probs[:, 1]
test_pred = (p_yes_test >= best["threshold"]).astype(int)
results["Neural Network (tuned threshold)"] = test_pred

test_acc  = accuracy_score(y_test, test_pred)
test_prec = precision_score(y_test, test_pred, pos_label=1, zero_division=0)
test_rec  = recall_score(y_test, test_pred, pos_label=1, zero_division=0)
test_f1   = f1_score(y_test, test_pred, pos_label=1, zero_division=0)
cm        = confusion_matrix(y_test, test_pred)

print("\nKeras NN Test Metrics (best run, tuned threshold):")
print(f"Accuracy:  {test_acc:.4f}")
print(f"Precision: {test_prec:.4f}")
print(f"Recall:    {test_rec:.4f}")
print(f"F1:        {test_f1:.4f}")
print("\nConfusion Matrix (rows=true, cols=pred):")
print(cm)



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









































































