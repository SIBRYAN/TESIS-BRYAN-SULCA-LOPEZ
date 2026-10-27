"""
02_limpieza_y_encoding.py  (adaptado a dataset_sipiera_500_real_y_sintetico.xlsx)
Lee la hoja Respuestas_Formulario (69 REAL + 431 SINTETICO), codifica y genera dataset_limpio.csv.
La columna origen_dato se conserva: el entrenamiento (03) usa SOLO las filas REAL; los sintéticos
se regeneran con SMOTE dentro de cada fold, para evitar fuga de datos.
"""
import pandas as pd

ARCHIVO = "dataset_sipiera_500_real_y_sintetico.xlsx"
df = pd.read_excel(ARCHIVO, sheet_name="Respuestas_Formulario")

RENOMBRAR = {
    "Marca temporal": "timestamp", "1. Ciclo académico actual": "ciclo",
    "2. Ingreso familiar mensual aproximado": "ingreso",
    "3. Facilidad de acceso a computadora e internet estable": "acceso_tec",
    "4. Nivel educativo más alto de tus padres o tutores": "nivel_edu_padres",
    "5. Colegio de procedencia": "colegio", "6. Distrito de residencia actual": "distrito",
    "7. ¿Trabajas actualmente mientras estudias?": "trabaja",
    "7a. ¿Cuántas horas semanales trabajas aproximadamente?": "horas_trabajo",
    "8. Promedio ponderado acumulado aproximado": "ppa",
    "9. ¿De qué ciclo(s) anteriores tienes cursos pendientes? (relativo al ciclo actual)": "ciclos_atraso",
    "10. ¿Cuántos cursos pendientes tienes en total?": "num_cursos_pendientes",
    "11. Rendimiento percibido en cursos críticos de primeros ciclos": "rendimiento_cursos_criticos",
    "12. Porcentaje aproximado de cursos aprobados en total": "pct_cursos_aprobados",
    "13. Puntaje o percentil en el examen de admisión": "puntaje_admision",
    "14. ¿Cómo describirías tu situación académica actual?": "situacion_academica",
}
df = df.rename(columns=RENOMBRAR)
print("Origen de los datos:", df["origen_dato"].value_counts().to_dict())
for c in df.select_dtypes("object").columns:
    df[c] = df[c].astype(str).str.strip().replace({"nan": pd.NA})

def ordinal(lista, desde=0): return {v: i + desde for i, v in enumerate(lista)}
map_ciclo = {"IX ciclo": 0, "X ciclo": 1}
map_ingreso = ordinal(["Menos de S/ 1,000", "S/ 1,000 – S/ 2,000", "S/ 2,001 – S/ 3,500", "S/ 3,501 – S/ 5,000", "Más de S/ 5,000"])
map_acceso = ordinal(["Muy difícil", "Difícil", "Regular", "Fácil", "Muy fácil"], 1)
map_edu = ordinal(["Primaria", "Secundaria", "Técnico", "Universitario", "Posgrado"])
map_horas = ordinal(["Menos de 10 h", "10 – 20 h", "21 – 30 h", "31 – 40 h", "Más de 40 h"], 1)   # vacío = 0
map_ppa = ordinal(["Menos de 11", "11 – 12.9", "13 – 14.9", "15 – 16.9", "17 – 20"])
map_rend = ordinal(["Muy bajo", "Bajo", "Regular", "Bueno", "Muy bueno"], 1)
map_pct = ordinal(["Menos de 60%", "60% – 74%", "75% – 89%", "90% – 100%"])
map_adm = ordinal(["Menos de 50", "50 – 69", "70 – 84", "85 – 100"])
map_target = {"En riesgo académico": 0, "Regular / estable": 1, "Buen rendimiento": 2}

df["ciclo_enc"] = df["ciclo"].map(map_ciclo)
df["ingreso_enc"] = df["ingreso"].map(map_ingreso)
df["acceso_tec"] = df["acceso_tec"].map(map_acceso)
df["nivel_edu_padres_enc"] = df["nivel_edu_padres"].map(map_edu)
df["trabaja_enc"] = df["trabaja"].map({"No": 0, "Sí": 1})
df["horas_trabajo_enc"] = df["horas_trabajo"].map(map_horas).fillna(0).astype(int)
df["ppa_enc"] = df["ppa"].map(map_ppa)
df["num_cursos_pendientes_enc"] = df["num_cursos_pendientes"].astype(float).astype(int)
df["rendimiento_cursos_criticos"] = df["rendimiento_cursos_criticos"].map(map_rend)
df["pct_cursos_aprobados_enc"] = df["pct_cursos_aprobados"].map(map_pct)
df["puntaje_admision_enc"] = df["puntaje_admision"].map(map_adm)
df["target"] = df["situacion_academica"].map(map_target)

txt = df["ciclos_atraso"].fillna("")
df["pend_ciclo_anterior"] = txt.str.contains("inmediato anterior").astype(int)
df["pend_2_ciclos"] = txt.str.contains("2 ciclos atrás").astype(int)
df["pend_3_o_mas_ciclos"] = txt.str.contains("3 ciclos a más").astype(int)
df["ciclos_atraso_enc"] = df["pend_ciclo_anterior"]
df.loc[df["pend_2_ciclos"] == 1, "ciclos_atraso_enc"] = 2
df.loc[df["pend_3_o_mas_ciclos"] == 1, "ciclos_atraso_enc"] = 3

cols_check = ["ciclo_enc", "ingreso_enc", "acceso_tec", "nivel_edu_padres_enc", "trabaja_enc", "ppa_enc",
              "rendimiento_cursos_criticos", "pct_cursos_aprobados_enc", "puntaje_admision_enc", "target", "colegio"]
nulos = df[cols_check].isna().sum()
assert nulos.sum() == 0, f"Hay valores sin mapear:\n{nulos[nulos > 0]}"

df_distrito = pd.get_dummies(df["distrito"], prefix="distrito").astype(int)
df_colegio = pd.get_dummies(df["colegio"].str.replace("Colegio ", ""), prefix="colegio").astype(int)
feats = ["ciclo_enc", "ingreso_enc", "acceso_tec", "nivel_edu_padres_enc", "trabaja_enc", "horas_trabajo_enc", "ppa_enc",
         "pend_ciclo_anterior", "pend_2_ciclos", "pend_3_o_mas_ciclos", "ciclos_atraso_enc", "num_cursos_pendientes_enc",
         "rendimiento_cursos_criticos", "pct_cursos_aprobados_enc", "puntaje_admision_enc"]
out = pd.concat([df[feats], df_colegio, df_distrito, df["target"].astype(int), df["origen_dato"]], axis=1)
out.to_csv("dataset_limpio.csv", index=False, encoding="utf-8-sig")
print(f"dataset_limpio.csv: {out.shape[0]} filas, {out.shape[1]} columnas")
print(out.groupby("origen_dato")["target"].value_counts().unstack().rename(columns={0: "En riesgo", 1: "Regular", 2: "Buen rendimiento"}))
