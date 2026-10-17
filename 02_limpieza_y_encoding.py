"""
02_limpieza_y_encoding.py  (adaptado al Excel dataset_sipiera_500_REAL.xlsx)
------------------------------------------------------------
Lee la hoja Respuestas_Formulario, limpia, codifica y genera:
    dataset_limpio.csv   (sin balancear: dashboard / Power BI y entrenamiento)

NOTA: el balanceo (SMOTE) ya NO se hace aquí. Se hace en el script 03
solo sobre el set de entrenamiento, para evitar fuga de datos
(si se balancea antes de separar train/test, hay registros sintéticos
casi idénticos a los de prueba y las métricas salen infladas).
------------------------------------------------------------
"""
import pandas as pd

ARCHIVO = "dataset_sipiera_500_REAL.xlsx"
df = pd.read_excel(ARCHIVO, sheet_name="Respuestas_Formulario")

RENOMBRAR = {
    "Marca temporal": "timestamp",
    "1. Ciclo académico actual": "ciclo",
    "2. Ingreso familiar mensual aproximado": "ingreso",
    "3. Facilidad de acceso a computadora e internet estable": "acceso_tec",
    "4. Nivel educativo más alto de tus padres o tutores": "nivel_edu_padres",
    "5. Colegio de procedencia": "colegio",
    "6. Distrito de residencia actual": "distrito",
    "7. ¿Trabajas actualmente mientras estudias?": "trabaja",
    "7a. ¿Cuántas horas semanales trabajas aproximadamente?": "horas_trabajo",
    "8. Promedio ponderado acumulado aproximado": "ppa",
    "9. ¿De qué ciclo(s) anteriores tienes cursos pendientes?": "ciclos_atraso",
    "10. ¿Cuántos cursos pendientes tienes en total?": "num_cursos_pendientes",
    "11. Rendimiento percibido en cursos críticos de primeros ciclos": "rendimiento_cursos_criticos",
    "12. Porcentaje aproximado de cursos aprobados en total": "pct_cursos_aprobados",
    "13. Puntaje o percentil en el examen de admisión": "puntaje_admision",
    "14. ¿Cómo describirías tu situación académica actual?": "situacion_academica",
}
df = df.rename(columns=RENOMBRAR)
print("Origen de los datos:", df["origen_dato"].unique().tolist())

# Normaliza espacios sobrantes en textos
for c in df.select_dtypes("object").columns:
    df[c] = df[c].astype(str).str.strip().replace({"nan": pd.NA})

# ---------------- Mapas ordinales ----------------
map_ciclo = {"9°": 0, "10°": 1, "11°": 2, "12° o más": 3}
map_ingreso = {"Menos de S/1000": 0, "S/1000 - S/2000": 1, "S/2000 - S/3000": 2,
               "S/3000 - S/4000": 3, "Más de S/4000": 4}
map_edu_padres = {"Primaria": 0, "Secundaria": 1, "Técnica": 2, "Universitaria": 3, "Posgrado": 4}
map_ppa = {"Menor a 11": 0, "11-12": 1, "12-13": 2, "13-14": 3, "14-15": 4, "Mayor a 15": 5}
map_num_pendientes = {"0": 0, "1": 1, "2": 2, "3": 3, "4 o más": 4}
map_pct_aprobados = {"Menos de 50%": 0, "50% - 70%": 1, "70% - 85%": 2, "85% - 95%": 3, "Más de 95%": 4}
map_puntaje_admision = {"Percentil bajo": 0, "Percentil medio": 1, "Percentil alto": 2,
                        "No recuerdo": 1}   # imputación neutral; se marca aparte con no_recuerda_admision
map_horas_trabajo = {"Menos de 10 horas": 1, "10-20 horas": 2, "20-30 horas": 3, "30 horas a más": 4}
map_target = {"Avance regular, sin atrasos": 2,
              "Avance con algún retraso (1 ciclo)": 1,
              "Riesgo de atraso significativo o posible abandono (2 o más ciclos)": 0}

df["ciclo_enc"] = df["ciclo"].map(map_ciclo)
df["ingreso_enc"] = df["ingreso"].map(map_ingreso)
df["nivel_edu_padres_enc"] = df["nivel_edu_padres"].map(map_edu_padres)
df["ppa_enc"] = df["ppa"].map(map_ppa)
df["num_cursos_pendientes_enc"] = df["num_cursos_pendientes"].map(map_num_pendientes)
df["pct_cursos_aprobados_enc"] = df["pct_cursos_aprobados"].map(map_pct_aprobados)
df["puntaje_admision_enc"] = df["puntaje_admision"].map(map_puntaje_admision)
df["no_recuerda_admision"] = (df["puntaje_admision"] == "No recuerdo").astype(int)
df["trabaja_enc"] = df["trabaja"].map({"No": 0, "Sí": 1, "Si": 1})
df["horas_trabajo_enc"] = df["horas_trabajo"].map(map_horas_trabajo).fillna(0).astype(int)  # vacío = no trabaja
df["colegio_privado"] = df["colegio"].map({"Publico": 0, "Público": 0, "Privado": 1})
df["target"] = df["situacion_academica"].map(map_target)

# Pregunta 9 es de casillas múltiples -> 3 columnas binarias
txt = df["ciclos_atraso"].fillna("")
df["pend_ciclo_anterior"] = txt.str.contains("inmediato anterior").astype(int)
df["pend_2_ciclos"] = txt.str.contains("2 ciclos atrás").astype(int)
df["pend_3_o_mas_ciclos"] = txt.str.contains("3 ciclos a más").astype(int)
# Para el gráfico del dashboard: nivel máximo de atraso (0 a 3)
df["ciclos_atraso_enc"] = (df["pend_ciclo_anterior"] * 1)
df.loc[df["pend_2_ciclos"] == 1, "ciclos_atraso_enc"] = 2
df.loc[df["pend_3_o_mas_ciclos"] == 1, "ciclos_atraso_enc"] = 3

# ---------------- Validación: que nada quede sin mapear ----------------
cols_check = ["ciclo_enc", "ingreso_enc", "nivel_edu_padres_enc", "ppa_enc", "num_cursos_pendientes_enc",
              "pct_cursos_aprobados_enc", "puntaje_admision_enc", "trabaja_enc", "colegio_privado", "target",
              "acceso_tec", "rendimiento_cursos_criticos"]
nulos = df[cols_check].isna().sum()
assert nulos.sum() == 0, f"Hay valores sin mapear:\n{nulos[nulos>0]}"

# ---------------- One-hot del distrito ----------------
df_distrito = pd.get_dummies(df["distrito"], prefix="distrito").astype(int)

columnas_features = [
    "ciclo_enc", "ingreso_enc", "acceso_tec", "nivel_edu_padres_enc", "colegio_privado",
    "trabaja_enc", "horas_trabajo_enc", "ppa_enc",
    "pend_ciclo_anterior", "pend_2_ciclos", "pend_3_o_mas_ciclos", "ciclos_atraso_enc",
    "num_cursos_pendientes_enc", "rendimiento_cursos_criticos",
    "pct_cursos_aprobados_enc", "puntaje_admision_enc", "no_recuerda_admision",
]
dataset_limpio = pd.concat([df[columnas_features], df_distrito, df["target"].astype(int)], axis=1)
dataset_limpio.to_csv("dataset_limpio.csv", index=False, encoding="utf-8-sig")
print(f"dataset_limpio.csv generado: {dataset_limpio.shape[0]} filas, {dataset_limpio.shape[1]} columnas")
print(dataset_limpio["target"].value_counts().rename({0: "Riesgo", 1: "Moderado", 2: "Alto"}))
