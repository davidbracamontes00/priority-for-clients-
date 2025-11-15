# -*- coding: utf-8 -*-
"""
===========================================
  LogisticMate - Análisis y Modelado Logístico
===========================================

Autor: David Bracamontes
Versión: 1.0
Fecha: 2025-11-10

Descripción:
------------
Script completo para análisis y modelado logístico usando Machine Learning.
Incluye:
- Limpieza y codificación de datos
- Visualizaciones exploratorias (EDA)
- Modelos supervisados (KNN, SVM, Random Forest, XGBoost)
- Segmentación por clustering optimizado (K-Means + silhouette)
- Función para predecir cluster de nuevos registros

Uso:
----
    python logisticmate_clean.py

Requisitos:
-----------
pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, openpyxl, joblib
"""

# =============================================================
#  PARTE 1 - Limpieza, Codificación y Exploración de Datos
# =============================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

print("\n=== PARTE 1: LIMPIEZA Y EDA ===\n")

# 1) Cargar dataset original
df = pd.read_excel("dataset1.xlsx")
df_encoded = df.copy()

# 2) Codificación de variables categóricas
categorical_cols = ['Warehouse_block', 'Product_importance', 'Mode_of_Shipment']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

df_encoded.to_excel("dataset_codificado.xlsx", index=False)
print("Codificación completada.\n")

# 3) Crear variable derivada útil
df['Vehiculo_eficiencia'] = (
    df['Mode_of_Shipment'].astype(str) + "_" +
    df['Reached.on.Time_Y.N'].map({1: "yes", 0: "no"})
)
df.to_excel("dataset_enriquecido.xlsx", index=False)

# 4) EDA – visualización rápida
df_viz = pd.read_excel("dataset_codificado.xlsx")
df_viz['Delivery_Status'] = df_viz['Reached.on.Time_Y.N'].map({1: 'On Time', 0: 'Delayed'})

sns.set(style="whitegrid")

plt.figure(figsize=(6, 4))
sns.countplot(data=df_viz, x='Delivery_Status')
plt.title("Entregas a tiempo vs retrasadas")
plt.show()

plt.figure(figsize=(7, 4))
sns.boxplot(data=df_viz, x='Delivery_Status', y='Discount_offered')
plt.title("Descuento vs Estado de entrega")
plt.show()

plt.figure(figsize=(7, 4))
sns.histplot(data=df_viz, x='Weight_in_gms', hue='Delivery_Status', kde=True, bins=30)
plt.title("Peso por estado de entrega")
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(df_viz.drop(columns='Delivery_Status').corr(), annot=True, cmap='coolwarm')
plt.title("Mapa de correlación")
plt.show()

print("EDA completado.\n")

# =============================================================
#  PARTE 2 - Modelos Supervisados
# =============================================================

print("\n=== PARTE 2: MODELOS SUPERVISADOS ===\n")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Cargar dataset final
df_ml = pd.read_excel("dataset_final.xlsx")

X = df_ml.drop(['Mode_of_Shipment', 'Vehiculo_eficiencia_code'], axis=1)
y = df_ml['Mode_of_Shipment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# --- KNN ---
print("Entrenando modelo KNN...")
knn = KNeighborsClassifier(n_neighbors=7, metric='manhattan')
knn.fit(X_train_s, y_train)
pred_knn = knn.predict(X_test_s)
print("\nKNN - Accuracy:", accuracy_score(y_test, pred_knn))
print(classification_report(y_test, pred_knn, target_names=["Flight","Road","Ship"]))

# --- SVM ---
print("\nEntrenando modelo SVM...")
df_bal = pd.read_csv("dataset_balanceado.csv")
X_svm = df_bal.drop(['Reached.on.Time_Y.N', 'Vehiculo_eficiencia_code'], axis=1)
y_svm = df_bal['Reached.on.Time_Y.N']

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_svm, y_svm, test_size=0.3, random_state=42)

scaler2 = StandardScaler()
X_train2_s = scaler2.fit_transform(X_train2)
X_test2_s = scaler2.transform(X_test2)

svm = SVC(kernel='rbf', C=10, gamma=0.1)
svm.fit(X_train2_s, y_train2)
pred_svm = svm.predict(X_test2_s)

print("\nSVM - Accuracy:", accuracy_score(y_test2, pred_svm))
print(classification_report(y_test2, pred_svm, target_names=["Delayed","On Time"]))

# --- Random Forest ---
print("\nEntrenando modelo Random Forest...")
df_rf = df_bal.copy()

X_rf = df_rf.drop(['Mode_of_Shipment','Vehiculo_eficiencia_code'], axis=1)
y_rf = df_rf['Mode_of_Shipment']

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.3, random_state=42)

scaler_rf = StandardScaler()
X_train_rf_s = scaler_rf.fit_transform(X_train_rf)
X_test_rf_s = scaler_rf.transform(X_test_rf)

rf = RandomForestClassifier(
    n_estimators=150, max_depth=15, min_samples_split=5,
    max_features='sqrt', bootstrap=True, n_jobs=-1, random_state=42
)
rf.fit(X_train_rf_s, y_train_rf)
pred_rf = rf.predict(X_test_rf_s)

print("\nRandom Forest - Accuracy:", accuracy_score(y_test_rf, pred_rf))
print(classification_report(y_test_rf, pred_rf, target_names=["Flight","Road","Ship"]))

# --- XGBoost ---
print("\nEntrenando modelo XGBoost...")
df_xgb = df_bal.copy()

X_xgb = df_xgb.drop(['Reached.on.Time_Y.N','Vehiculo_eficiencia_code'], axis=1)
y_xgb = df_xgb['Reached.on.Time_Y.N']

X_train_x, X_test_x, y_train_x, y_test_x = train_test_split(
    X_xgb, y_xgb, test_size=0.3, random_state=42
)

xgb_model = xgb.XGBClassifier(
    objective='binary:logistic', eval_metric='logloss',
    max_depth=5, learning_rate=0.1, n_estimators=100
)

xgb_model.fit(X_train_x, y_train_x)
pred_xgb = xgb_model.predict(X_test_x)

print("\nXGBoost - Accuracy:", accuracy_score(y_test_x, pred_xgb))
print(classification_report(y_test_x, pred_xgb, target_names=["Delayed","On Time"]))

print("\nModelos supervisados completados.\n")

# =============================================================
#  PARTE 3 - Clustering (KMeans Optimizado)
# =============================================================

print("\n=== PARTE 3: KMEANS OPTIMIZADO ===\n")

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import joblib

df_clust = pd.read_excel("dataset_final.xlsx")

# Variables numéricas útiles
cols_excluir = {'Mode_of_Shipment', 'Vehiculo_eficiencia_code'}
num_cols = [c for c in df_clust.select_dtypes(include=[np.number]).columns if c not in cols_excluir]

X = df_clust[num_cols].copy()

# Pipeline: imputación + escalado
preprocess = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

X_pre = preprocess.fit_transform(X)

# Determinar k óptimo (silhouette + codo)
K_range = range(2, 10)
sil_scores = []

for k in K_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_pre)
    sil = silhouette_score(X_pre, labels)
    sil_scores.append(sil)

k_opt = K_range[np.argmax(sil_scores)]
print(f"k óptimo según silhouette: {k_opt}")

# Entrenar KMeans final
kmeans = KMeans(n_clusters=k_opt, n_init=10, random_state=42)
df_clust["Cluster"] = kmeans.fit_predict(X_pre)

df_clust.to_excel("resultados_con_clusters.xlsx", index=False)
joblib.dump(kmeans, "kmeans_model.pkl")
joblib.dump(preprocess, "kmeans_preprocess.pkl")

print("Modelo KMeans entrenado y guardado.\n")

# Función para predecir cluster de nuevos registros
def predecir_cluster(registro: dict) -> int:
    """
    registro: diccionario con columnas numéricas originales
    """
    df_new = pd.DataFrame([registro])
    X_new = preprocess.transform(df_new)
    return int(kmeans.predict(X_new)[0])

# Ejemplo de predicción
ejemplo = {col: df_clust[col].median() for col in num_cols}
pred = predecir_cluster(ejemplo)
print(f"Predicción ejemplo → Cluster {pred}")

print("\n=== LogisticMate Finalizado ===")
