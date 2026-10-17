"""
03_entrenamiento_modelos.py  (versión alineada a la tesis)
------------------------------------------------------------
Entrena y compara Random Forest, XGBoost y una Red Neuronal
Artificial en TensorFlow/Keras, sobre dataset_limpio.csv.
Genera gráficos con Matplotlib/Seaborn (como indica la sección
3.5 de la tesis) y guarda resultados para el dashboard Streamlit.

ENTRADA:  dataset_limpio.csv (SMOTE se aplica aquí, solo al set de entrenamiento)
SALIDA:   resultados_metricas.csv
          feature_importance.csv
          predicciones_test.csv
          graficos/*.png   (matriz de confusión, comparación de
                             métricas, feature importance)
          modelo_red_neuronal.keras
------------------------------------------------------------
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # para poder generar imágenes sin pantalla
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTENC

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import shap
import joblib

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

os.makedirs("graficos", exist_ok=True)
sns.set_theme(style="whitegrid")
NOMBRES_CLASES = ["Riesgo", "Moderado", "Alto"]

# ---------------------------------------------------------------
# 1. CARGA Y SPLIT
# ---------------------------------------------------------------
df = pd.read_csv("dataset_limpio.csv")
X = df.drop(columns=["target"])
y = df["target"]

# 1) Primero se separa train/test (el test NUNCA se toca con SMOTE)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print("Train antes de SMOTE:", y_train.value_counts().sort_index().to_dict())

# 2) SMOTE-NC solo sobre entrenamiento. Columnas binarias/one-hot = categóricas
cols_cat = [c for c in X.columns if X[c].nunique() <= 2]
cat_idx = [X.columns.get_loc(c) for c in cols_cat]
k = max(1, min(5, y_train.value_counts().min() - 1))
smote = SMOTENC(categorical_features=cat_idx, random_state=RANDOM_STATE, k_neighbors=k)
X_train, y_train = smote.fit_resample(X_train, y_train)
print("Train después de SMOTE:", y_train.value_counts().sort_index().to_dict())

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# One-hot del target, necesario para la salida softmax de Keras
y_train_cat = keras.utils.to_categorical(y_train, num_classes=3)
y_test_cat = keras.utils.to_categorical(y_test, num_classes=3)

# ---------------------------------------------------------------
# 2. RANDOM FOREST
# ---------------------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced"
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)

# ---------------------------------------------------------------
# 3. XGBOOST
# ---------------------------------------------------------------
xgb = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    random_state=RANDOM_STATE, eval_metric="mlogloss"
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
y_proba_xgb = xgb.predict_proba(X_test)

# ---------------------------------------------------------------
# 4. RED NEURONAL ARTIFICIAL (TensorFlow/Keras)
# ---------------------------------------------------------------
modelo_rna = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(3, activation="softmax"),
])

modelo_rna.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=15, restore_best_weights=True
)

historia = modelo_rna.fit(
    X_train_scaled, y_train_cat,
    validation_split=0.2,
    epochs=200, batch_size=16,
    callbacks=[early_stop],
    verbose=0
)

y_proba_rna = modelo_rna.predict(X_test_scaled, verbose=0)
y_pred_rna = np.argmax(y_proba_rna, axis=1)

modelo_rna.save("modelo_red_neuronal.keras")

# Gráfico de curva de entrenamiento (accuracy y loss por época)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(historia.history["accuracy"], label="Entrenamiento")
axes[0].plot(historia.history["val_accuracy"], label="Validación")
axes[0].set_title("Accuracy por época - Red Neuronal")
axes[0].set_xlabel("Época")
axes[0].legend()
axes[1].plot(historia.history["loss"], label="Entrenamiento")
axes[1].plot(historia.history["val_loss"], label="Validación")
axes[1].set_title("Loss por época - Red Neuronal")
axes[1].set_xlabel("Época")
axes[1].legend()
plt.tight_layout()
plt.savefig("graficos/curva_entrenamiento_rna.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 5. MÉTRICAS COMPARATIVAS
# ---------------------------------------------------------------
def calcular_metricas(nombre, y_true, y_pred, y_proba):
    return {
        "Modelo": nombre,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1-score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "AUC-ROC": roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"),
        "Recall_clase_Riesgo": recall_score(y_true, y_pred, labels=[0], average="macro", zero_division=0),
    }

resultados = [
    calcular_metricas("Random Forest", y_test, y_pred_rf, y_proba_rf),
    calcular_metricas("XGBoost", y_test, y_pred_xgb, y_proba_xgb),
    calcular_metricas("Red Neuronal (Keras)", y_test, y_pred_rna, y_proba_rna),
]
resultados_df = pd.DataFrame(resultados).sort_values("F1-score", ascending=False)
resultados_df.to_csv("resultados_metricas.csv", index=False, encoding="utf-8-sig")

print("=== COMPARACIÓN FINAL ===")
print(resultados_df.to_string(index=False))

for nombre, y_true, y_pred in [
    ("Random Forest", y_test, y_pred_rf),
    ("XGBoost", y_test, y_pred_xgb),
    ("Red Neuronal (Keras)", y_test, y_pred_rna),
]:
    print(f"\n=== {nombre} ===")
    print(classification_report(y_true, y_pred, target_names=NOMBRES_CLASES, zero_division=0))

# ---------------------------------------------------------------
# 6. GRÁFICOS: matrices de confusión (Seaborn heatmap) y comparación de métricas
# ---------------------------------------------------------------
matrices = {
    "Random Forest": confusion_matrix(y_test, y_pred_rf),
    "XGBoost": confusion_matrix(y_test, y_pred_xgb),
    "Red Neuronal (Keras)": confusion_matrix(y_test, y_pred_rna),
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (nombre, cm) in zip(axes, matrices.items()):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=NOMBRES_CLASES, yticklabels=NOMBRES_CLASES)
    ax.set_title(nombre)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
plt.tight_layout()
plt.savefig("graficos/matrices_confusion.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
metricas_plot = resultados_df.melt(
    id_vars="Modelo", value_vars=["Accuracy", "Precision", "Recall", "F1-score"],
    var_name="Métrica", value_name="Valor"
)
sns.barplot(data=metricas_plot, x="Métrica", y="Valor", hue="Modelo", ax=ax)
ax.set_ylim(0, 1.05)
ax.set_title("Comparación de métricas por modelo")
plt.tight_layout()
plt.savefig("graficos/comparacion_metricas.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 7. FEATURE IMPORTANCE (RF y XGBoost)
# ---------------------------------------------------------------
fi_rf = pd.Series(rf.feature_importances_, index=X.columns, name="RandomForest")
fi_xgb = pd.Series(xgb.feature_importances_, index=X.columns, name="XGBoost")
feature_importance = pd.concat([fi_rf, fi_xgb], axis=1).sort_values("RandomForest", ascending=False)
feature_importance.to_csv("feature_importance.csv", encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(8, 6))
feature_importance.head(10).plot(kind="barh", ax=ax)
ax.invert_yaxis()
ax.set_title("Top 10 variables más influyentes")
plt.tight_layout()
plt.savefig("graficos/feature_importance.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 8. SHAP sobre el mejor modelo
# ---------------------------------------------------------------
mejor_modelo_nombre = resultados_df.iloc[0]["Modelo"]
print(f"\nCalculando SHAP para el mejor modelo: {mejor_modelo_nombre}")

try:
    if mejor_modelo_nombre == "Random Forest":
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_test)
    elif mejor_modelo_nombre == "XGBoost":
        explainer = shap.TreeExplainer(xgb)
        shap_values = explainer.shap_values(X_test)
    else:
        explainer = shap.KernelExplainer(modelo_rna.predict, X_train_scaled[:50])
        shap_values = explainer.shap_values(X_test_scaled[:30])

    if isinstance(shap_values, list):
        sv = np.mean([np.abs(v) for v in shap_values], axis=0)   # lista por clase
    else:
        sv = np.abs(shap_values)
        if sv.ndim == 3:
            sv = sv.mean(axis=2)                                  # (muestras, vars, clases)
    shap_abs_mean = sv.mean(axis=0)
    shap_summary = pd.Series(np.ravel(shap_abs_mean)[:len(X.columns)], index=X.columns, name="SHAP_importancia_media")
    shap_summary.sort_values(ascending=False).to_csv("shap_importancia.csv", encoding="utf-8-sig")
    print("shap_importancia.csv generado")
except Exception as e:
    print(f"Nota: SHAP no se pudo calcular automáticamente ({e})")

# ---------------------------------------------------------------
# 9. EXPORTAR PREDICCIONES (para el dashboard Streamlit)
# ---------------------------------------------------------------
predicciones = {
    "Random Forest": y_pred_rf,
    "XGBoost": y_pred_xgb,
    "Red Neuronal (Keras)": y_pred_rna,
}
mejor_pred = predicciones[mejor_modelo_nombre]

X_test_out = X_test.copy()
X_test_out["real"] = y_test.values
X_test_out["prediccion"] = mejor_pred
X_test_out["modelo_usado"] = mejor_modelo_nombre
# Predicciones de los 3 modelos (el dashboard permite compararlos)
X_test_out["pred_rf"] = y_pred_rf
X_test_out["pred_xgb"] = y_pred_xgb
X_test_out["pred_rna"] = y_pred_rna
X_test_out.to_csv("predicciones_test.csv", index=False, encoding="utf-8-sig")

# Modelo Random Forest guardado para el simulador del dashboard
joblib.dump({"modelo": rf, "columnas": list(X.columns)}, "modelo_random_forest.joblib")

print("\nListo: resultados_metricas.csv, feature_importance.csv, predicciones_test.csv,")
print("modelo_red_neuronal.keras y gráficos en /graficos/ generados correctamente.")
