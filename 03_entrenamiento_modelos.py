"""
03_entrenamiento_modelos.py  (evaluación honesta con 69 datos reales)
------------------------------------------------------------------
Compara Random Forest, XGBoost y Red Neuronal (Keras) usando SOLO las filas REAL de dataset_limpio.csv.
Evaluación: validación cruzada estratificada 5 pliegues x 2 repeticiones. SMOTE-NC se aplica DENTRO de cada
pliegue y solo al entrenamiento, así ningún dato sintético se parece a los datos de prueba (sin fuga).
Los 431 sintéticos del Excel sirven para exploración/dashboard, no para evaluar.
SALIDA: resultados_metricas.csv, feature_importance.csv, predicciones_test.csv (predicciones fuera de
        muestra de los 69 reales), shap_importancia.csv, modelo_random_forest.joblib,
        modelo_red_neuronal.keras, graficos/*.png
"""
import os, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTENC
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import shap, joblib
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)
os.makedirs("graficos", exist_ok=True)
sns.set_theme(style="whitegrid")
CLASES = ["En riesgo", "Regular", "Buen rendimiento"]

df = pd.read_csv("dataset_limpio.csv")
real = df[df["origen_dato"] == "REAL"].reset_index(drop=True)
X = real.drop(columns=["target", "origen_dato"]); y = real["target"].values
cat_idx = [i for i, c in enumerate(X.columns) if X[c].nunique() <= 2]
print(f"Evaluando con {len(X)} estudiantes REALES:", dict(zip(CLASES, np.bincount(y))))

def balancear(Xt, yt):
    k = max(1, min(5, np.bincount(yt).min() - 1))
    return SMOTENC(categorical_features=cat_idx, k_neighbors=k, random_state=SEED).fit_resample(Xt, yt)

def modelo_rna(n):
    m = keras.Sequential([layers.Input(shape=(n,)), layers.Dense(64, activation="relu"), layers.Dropout(0.3),
                          layers.Dense(32, activation="relu"), layers.Dropout(0.2), layers.Dense(3, activation="softmax")])
    m.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return m

def fit_predict(Xt, yt, Xv):
    """Entrena los 3 modelos con (Xt, yt) ya balanceado y devuelve probabilidades sobre Xv."""
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, class_weight="balanced").fit(Xt, yt)
    xg = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, random_state=SEED, eval_metric="mlogloss").fit(Xt, yt)
    sc = StandardScaler().fit(Xt)
    nn = modelo_rna(Xt.shape[1])
    nn.fit(sc.transform(Xt), keras.utils.to_categorical(yt, 3), validation_split=0.2, epochs=200, batch_size=16, verbose=0,
           callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)])
    return {"Random Forest": rf.predict_proba(Xv), "XGBoost": xg.predict_proba(Xv),
            "Red Neuronal (Keras)": nn.predict(sc.transform(Xv), verbose=0)}

def metricas(yt, proba):
    p = proba.argmax(1)
    return {"Accuracy": accuracy_score(yt, p),
            "Precision": precision_score(yt, p, average="weighted", zero_division=0),
            "Recall": recall_score(yt, p, average="weighted", zero_division=0),
            "F1-score": f1_score(yt, p, average="weighted", zero_division=0),
            "AUC-ROC": roc_auc_score(yt, proba, multi_class="ovr", average="weighted"),
            "Recall_clase_Riesgo": recall_score(yt, p, labels=[0], average="macro", zero_division=0)}

# ---------------- VALIDACIÓN CRUZADA (SMOTE solo dentro del entrenamiento) ----------------
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
filas = {m: [] for m in ["Random Forest", "XGBoost", "Red Neuronal (Keras)"]}
oof = {m: np.zeros((len(X), 3)) for m in filas}          # predicciones fuera de muestra (repetición 1)
for i, (tr, te) in enumerate(cv.split(X, y)):
    Xb, yb = balancear(X.iloc[tr], y[tr])
    prob = fit_predict(Xb, yb, X.iloc[te])
    for m, p in prob.items():
        filas[m].append(metricas(y[te], p))
        if i < 5: oof[m][te] = p
    print(f"pliegue {i + 1}/10 listo")

res = []
for m, lst in filas.items():
    d = pd.DataFrame(lst); fila = {"Modelo": m}
    for c in d.columns: fila[c] = d[c].mean(); fila[c + "_std"] = d[c].std()
    res.append(fila)
resultados = pd.DataFrame(res).sort_values("F1-score", ascending=False)
resultados.to_csv("resultados_metricas.csv", index=False, encoding="utf-8-sig")
print("\n=== COMPARACIÓN (media de 10 pliegues) ===")
print(resultados[["Modelo", "Accuracy", "F1-score", "AUC-ROC", "Recall_clase_Riesgo"]].round(3).to_string(index=False))
mejor = resultados.iloc[0]["Modelo"]

pred = X.copy(); pred["real"] = y
for k, m in zip(["pred_rf", "pred_xgb", "pred_rna"], filas):
    pred[k] = oof[m].argmax(1)
    pred["proba_" + k[5:]] = oof[m].max(1)
pred["prediccion"] = pred[{"Random Forest": "pred_rf", "XGBoost": "pred_xgb", "Red Neuronal (Keras)": "pred_rna"}[mejor]]
pred["modelo_usado"] = mejor
pred.to_csv("predicciones_test.csv", index=False, encoding="utf-8-sig")
for m, k in [("Random Forest", "pred_rf"), ("XGBoost", "pred_xgb"), ("Red Neuronal (Keras)", "pred_rna")]:
    print(f"\n=== {m} (predicciones fuera de muestra, 69 reales) ===")
    print(classification_report(y, pred[k], target_names=CLASES, zero_division=0))

# ---------------- GRÁFICOS ----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (m, k) in zip(axes, [("Random Forest", "pred_rf"), ("XGBoost", "pred_xgb"), ("Red Neuronal (Keras)", "pred_rna")]):
    sns.heatmap(confusion_matrix(y, pred[k]), annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=CLASES, yticklabels=CLASES)
    ax.set_title(m); ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
plt.tight_layout(); plt.savefig("graficos/matrices_confusion.png", dpi=150); plt.close()
mp = resultados.melt(id_vars="Modelo", value_vars=["Accuracy", "Precision", "Recall", "F1-score"], var_name="Métrica", value_name="Valor")
fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(data=mp, x="Métrica", y="Valor", hue="Modelo", ax=ax)
ax.set_ylim(0, 1.05); ax.set_title("Comparación de métricas (validación cruzada, 69 reales)")
plt.tight_layout(); plt.savefig("graficos/comparacion_metricas.png", dpi=150); plt.close()

# ---------------- MODELOS FINALES (todos los reales + SMOTE) ----------------
Xb, yb = balancear(X, y)
rf = RandomForestClassifier(n_estimators=300, random_state=SEED, class_weight="balanced").fit(Xb, yb)
xg = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, random_state=SEED, eval_metric="mlogloss").fit(Xb, yb)
sc = StandardScaler().fit(Xb); nn = modelo_rna(Xb.shape[1])
h = nn.fit(sc.transform(Xb), keras.utils.to_categorical(yb, 3), validation_split=0.2, epochs=200, batch_size=16, verbose=0,
           callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)])
nn.save("modelo_red_neuronal.keras")
joblib.dump({"modelo": rf, "columnas": list(X.columns)}, "modelo_random_forest.joblib")
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for ax, k, t in zip(axs, ["accuracy", "loss"], ["Accuracy", "Loss"]):
    ax.plot(h.history[k], label="Entrenamiento"); ax.plot(h.history["val_" + k], label="Validación"); ax.set_title(t + " por época - Red Neuronal"); ax.legend()
plt.tight_layout(); plt.savefig("graficos/curva_entrenamiento_rna.png", dpi=150); plt.close()

fi = pd.concat([pd.Series(rf.feature_importances_, index=X.columns, name="RandomForest"),
                pd.Series(xg.feature_importances_, index=X.columns, name="XGBoost")], axis=1).sort_values("RandomForest", ascending=False)
fi.to_csv("feature_importance.csv", encoding="utf-8-sig")
fig, ax = plt.subplots(figsize=(8, 6)); fi.head(10).plot(kind="barh", ax=ax); ax.invert_yaxis(); ax.set_title("Top 10 variables más influyentes")
plt.tight_layout(); plt.savefig("graficos/feature_importance.png", dpi=150); plt.close()

# SHAP sobre el Random Forest final (evaluado en los 69 reales)
try:
    sv = shap.TreeExplainer(rf).shap_values(X)
    sv = np.mean([np.abs(v) for v in sv], axis=0) if isinstance(sv, list) else (np.abs(sv).mean(axis=2) if sv.ndim == 3 else np.abs(sv))
    pd.Series(sv.mean(axis=0), index=X.columns, name="SHAP_importancia_media").sort_values(ascending=False).to_csv("shap_importancia.csv", encoding="utf-8-sig")
    print("shap_importancia.csv generado")
except Exception as e:
    print("SHAP no se pudo calcular:", e)
print("\nListo. Mejor modelo por F1:", mejor)
