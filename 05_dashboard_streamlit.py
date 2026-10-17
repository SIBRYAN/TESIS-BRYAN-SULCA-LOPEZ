"""
05_dashboard_streamlit.py
------------------------------------------------------------
SIPIERA-FISI: Sistema de predicción de éxito académico
Tesis: Desarrollo y Evaluación de un Modelo de Machine Learning para la
Predicción del Éxito Académico basado en Variables Socioeconómicas y
Académicas de Estudiantes de la FISI.
Autor y tesista: Bryan Sulca Lopez

EJECUTAR:   streamlit run 05_dashboard_streamlit.py
REQUIERE:   haber corrido 02 y 03 en esta carpeta, y tener junto a este
            archivo: dataset_sipiera_500_REAL.xlsx
------------------------------------------------------------
"""
import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

AUTOR = "Bryan Sulca Lopez"
EXCEL = "dataset_sipiera_500_REAL.xlsx"
CLASES = ["Riesgo", "Moderado", "Alto"]           # 0, 1, 2
COLOR = {"Riesgo": "#FF6B6B", "Moderado": "#FFC857", "Alto": "#3DDC97"}
AZUL, TEXTO, APAGADO, SUPERFICIE = "#5B8DEF", "#E6EDF7", "#8FA0BA", "#131C2E"
MAPA_TARGET = {0: "Riesgo", 1: "Moderado", 2: "Alto"}
MODELOS = {"Random Forest": "pred_rf", "XGBoost": "pred_xgb", "Red Neuronal (Keras)": "pred_rna"}

st.set_page_config(page_title="SIPIERA-FISI", page_icon="🎓", layout="wide")

# ---------------------------------------------------------------
# ESTILO
# ---------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, button, input {{ font-family: 'Manrope', sans-serif !important; }}
.block-container {{ padding-top: 1.6rem; max-width: 1280px; }}
.hero {{ background: linear-gradient(120deg, #16233f 0%, #0f1a30 55%, #0B1220 100%);
        border: 1px solid #22314f; border-radius: 22px; padding: 34px 38px; margin-bottom: 22px; }}
.hero h1 {{ font-size: 3rem; font-weight: 800; letter-spacing: -0.03em; margin: 0 0 6px 0; color: {TEXTO}; }}
.hero p.tesis {{ color: {APAGADO}; font-size: 1.02rem; max-width: 820px; line-height: 1.55; margin: 0 0 18px 0; }}
.chip {{ display: inline-block; padding: 7px 15px; border-radius: 999px; font-size: .86rem; font-weight: 600;
        margin: 0 8px 6px 0; border: 1px solid #2d3f63; background: #0f1a30; color: {TEXTO}; }}
.chip.autor {{ background: {AZUL}; border-color: {AZUL}; color: #fff; }}
.kpi {{ background: {SUPERFICIE}; border: 1px solid #22314f; border-radius: 16px; padding: 18px 20px; height: 100%; }}
.kpi .v {{ font-size: 2.2rem; font-weight: 800; line-height: 1.1; color: {TEXTO}; }}
.kpi .l {{ color: {APAGADO}; font-size: .88rem; margin-top: 4px; }}
.resultado {{ border-radius: 18px; padding: 22px 26px; border: 1px solid #22314f; background: {SUPERFICIE}; }}
.resultado .v {{ font-size: 2.3rem; font-weight: 800; }}
.footer {{ color: {APAGADO}; text-align: center; font-size: .85rem; margin-top: 38px; padding-top: 18px; border-top: 1px solid #22314f; }}
div[data-baseweb="tab-list"] {{ gap: 6px; }}
button[data-baseweb="tab"] {{ border-radius: 10px 10px 0 0; padding: 10px 18px; font-weight: 600; }}
section[data-testid="stSidebar"] {{ border-right: 1px solid #22314f; }}
</style>
""", unsafe_allow_html=True)


def estilo(fig, alto=380):
    fig.update_layout(
        height=alto, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Manrope, sans-serif", color=TEXTO),
        margin=dict(l=10, r=10, t=50, b=10), legend_title_text="",
        title_font=dict(size=16),
    )
    fig.update_xaxes(gridcolor="#1d2a45", zerolinecolor="#1d2a45")
    fig.update_yaxes(gridcolor="#1d2a45", zerolinecolor="#1d2a45")
    return fig


# ---------------------------------------------------------------
# CARGA DE DATOS
# ---------------------------------------------------------------
RENOMBRAR = {
    "1. Ciclo académico actual": "Ciclo",
    "2. Ingreso familiar mensual aproximado": "Ingreso familiar",
    "3. Facilidad de acceso a computadora e internet estable": "Acceso tecnológico (1-5)",
    "4. Nivel educativo más alto de tus padres o tutores": "Educación de los padres",
    "5. Colegio de procedencia": "Colegio",
    "6. Distrito de residencia actual": "Distrito",
    "7. ¿Trabajas actualmente mientras estudias?": "Trabaja",
    "7a. ¿Cuántas horas semanales trabajas aproximadamente?": "Horas de trabajo",
    "8. Promedio ponderado acumulado aproximado": "Promedio ponderado",
    "9. ¿De qué ciclo(s) anteriores tienes cursos pendientes?": "Ciclos con pendientes",
    "10. ¿Cuántos cursos pendientes tienes en total?": "Cursos pendientes",
    "11. Rendimiento percibido en cursos críticos de primeros ciclos": "Rendimiento en cursos críticos (1-5)",
    "12. Porcentaje aproximado de cursos aprobados en total": "% de cursos aprobados",
    "13. Puntaje o percentil en el examen de admisión": "Percentil de admisión",
    "14. ¿Cómo describirías tu situación académica actual?": "Situación académica",
}
ORDEN = {
    "Ciclo": ["9°", "10°"],
    "Ingreso familiar": ["Menos de S/1000", "S/1000 - S/2000", "S/2000 - S/3000", "S/3000 - S/4000", "Más de S/4000"],
    "Educación de los padres": ["Primaria", "Secundaria", "Técnica", "Universitaria", "Posgrado"],
    "Horas de trabajo": ["No trabaja", "Menos de 10 horas", "10-20 horas", "20-30 horas", "30 horas a más"],
    "Promedio ponderado": ["Menor a 11", "11-12", "12-13", "13-14", "14-15", "Mayor a 15"],
    "Cursos pendientes": ["0", "1", "2", "3", "4 o más"],
    "% de cursos aprobados": ["Menos de 50%", "50% - 70%", "70% - 85%", "85% - 95%", "Más de 95%"],
    "Percentil de admisión": ["Percentil bajo", "Percentil medio", "Percentil alto", "No recuerdo"],
    "Situación académica": ["Riesgo", "Moderado", "Alto"],
}
SIT_ETIQUETA = {
    "Avance regular, sin atrasos": "Alto",
    "Avance con algún retraso (1 ciclo)": "Moderado",
    "Riesgo de atraso significativo o posible abandono (2 o más ciclos)": "Riesgo",
}


@st.cache_data
def cargar_excel():
    d = pd.read_excel(EXCEL, sheet_name="Respuestas_Formulario").rename(columns=RENOMBRAR)
    d["Horas de trabajo"] = d["Horas de trabajo"].fillna("No trabaja")
    d["Trabaja"] = d["Trabaja"].replace({"Si": "Sí"})
    d["Ciclos con pendientes"] = d["Ciclos con pendientes"].fillna("Ninguno")
    d["Cursos pendientes"] = d["Cursos pendientes"].astype(str)
    d["Situación académica"] = d["Situación académica"].map(SIT_ETIQUETA)
    return d


@st.cache_data
def cargar_csv(ruta, **kw):
    return pd.read_csv(ruta, **kw)


try:
    raw = cargar_excel()
    limpio = cargar_csv("dataset_limpio.csv")
    resultados = cargar_csv("resultados_metricas.csv")
    fi = cargar_csv("feature_importance.csv", index_col=0)
    pred = cargar_csv("predicciones_test.csv")
except FileNotFoundError as e:
    st.error(f"Falta un archivo: **{e.filename}**. Corre primero 02_limpieza_y_encoding.py y "
             f"03_entrenamiento_modelos.py, y deja el Excel en esta misma carpeta.")
    st.stop()

shap_df = cargar_csv("shap_importancia.csv", index_col=0) if os.path.exists("shap_importancia.csv") else None
origen = raw["origen_dato"].iloc[0] if "origen_dato" in raw else "REAL"

# ---------------------------------------------------------------
# CABECERA
# ---------------------------------------------------------------
st.markdown(f"""
<div class="hero">
  <h1>SIPIERA-FISI</h1>
  <p class="tesis">Desarrollo y Evaluación de un Modelo de Machine Learning para la Predicción del Éxito Académico
  basado en Variables Socioeconómicas y Académicas de Estudiantes de la FISI.</p>
  <span class="chip autor">Autor y tesista: {AUTOR}</span>
  <span class="chip">Facultad de Ingeniería de Sistemas e Informática, UNMSM</span>
  <span class="chip">{len(raw)} estudiantes encuestados</span>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# FILTROS (barra lateral, selección múltiple)
# ---------------------------------------------------------------
st.sidebar.markdown("### Filtros")
st.sidebar.caption("Aplican a la pestaña Panorama y a la descarga de datos.")
f_ciclo = st.sidebar.multiselect("Ciclo", ORDEN["Ciclo"], default=ORDEN["Ciclo"])
f_sit = st.sidebar.multiselect("Situación académica", CLASES, default=CLASES)
f_dist = st.sidebar.multiselect("Distrito", sorted(raw["Distrito"].unique()), placeholder="Todos los distritos")
f_cole = st.sidebar.multiselect("Colegio", sorted(raw["Colegio"].unique()), default=sorted(raw["Colegio"].unique()))
f_trab = st.sidebar.multiselect("¿Trabaja?", ["Sí", "No"], default=["Sí", "No"])
f_ing = st.sidebar.multiselect("Ingreso familiar", ORDEN["Ingreso familiar"], placeholder="Todos los ingresos")

df = raw[
    raw["Ciclo"].isin(f_ciclo) & raw["Situación académica"].isin(f_sit) &
    raw["Colegio"].isin(f_cole) & raw["Trabaja"].isin(f_trab)
]
if f_ing:
    df = df[df["Ingreso familiar"].isin(f_ing)]
if f_dist:
    df = df[df["Distrito"].isin(f_dist)]
st.sidebar.markdown(f"**{len(df)}** de {len(raw)} estudiantes seleccionados")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Panorama", "Comparación de modelos", "Variables influyentes", "Simulador de riesgo", "Casos y reportes"])

# ===============================================================
# TAB 1: PANORAMA
# ===============================================================
with tab1:
    if df.empty:
        st.warning("Los filtros no dejan ningún estudiante. Amplía la selección en la barra lateral.")
    else:
        pct_riesgo = (df["Situación académica"] == "Riesgo").mean() * 100
        pct_trab = (df["Trabaja"] == "Sí").mean() * 100
        pct_pub = (df["Colegio"] == "Publico").mean() * 100
        ppa_moda = df["Promedio ponderado"].mode().iloc[0]
        k = st.columns(4)
        for col, v, l in zip(k, [f"{len(df)}", f"{pct_riesgo:.0f}%", f"{pct_trab:.0f}%", ppa_moda],
                             ["Estudiantes en la selección", "En riesgo de atraso o abandono",
                              "Trabajan mientras estudian", "Promedio ponderado más frecuente"]):
            col.markdown(f'<div class="kpi"><div class="v">{v}</div><div class="l">{l}</div></div>', unsafe_allow_html=True)
        st.write("")

        c1, c2 = st.columns([1, 1.4])
        with c1:
            cnt = df["Situación académica"].value_counts().reindex(CLASES).fillna(0).reset_index()
            cnt.columns = ["Situación", "n"]
            fig = px.pie(cnt, names="Situación", values="n", hole=0.62, color="Situación",
                         color_discrete_map=COLOR, title="Nivel de éxito académico")
            fig.update_traces(textinfo="percent", textfont_size=14, marker=dict(line=dict(color="#0B1220", width=3)))
            st.plotly_chart(estilo(fig, 380), use_container_width=True)
        with c2:
            dd = (df.groupby("Distrito")["Situación académica"]
                  .apply(lambda s: (s == "Riesgo").mean() * 100).reset_index(name="% en riesgo"))
            dd["n"] = df.groupby("Distrito").size().values
            dd = dd.sort_values("% en riesgo").tail(12)
            fig = px.bar(dd, x="% en riesgo", y="Distrito", orientation="h", hover_data=["n"],
                         title="Distritos con mayor % de estudiantes en riesgo (top 12)",
                         color="% en riesgo", color_continuous_scale=["#2d3f63", "#FF6B6B"])
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(estilo(fig, 380), use_container_width=True)

        st.markdown("#### Compara variables contra la situación académica")
        variables = [v for v in ORDEN if v != "Situación académica"] + ["Distrito", "Colegio", "Trabaja",
                                                                         "Acceso tecnológico (1-5)",
                                                                         "Rendimiento en cursos críticos (1-5)"]
        o1, o2 = st.columns([3, 1])
        elegidas = o1.multiselect("Variables a graficar (elige varias)", variables,
                                  default=["Promedio ponderado", "% de cursos aprobados"])
        modo = o2.radio("Mostrar", ["Porcentaje", "Cantidad"], horizontal=True)
        cols = st.columns(2)
        for i, var in enumerate(elegidas):
            t = df.groupby([var, "Situación académica"]).size().reset_index(name="n")
            if modo == "Porcentaje":
                t["n"] = t["n"] / t.groupby(var)["n"].transform("sum") * 100
            orden_cat = ORDEN.get(var) or sorted(df[var].astype(str).unique())
            t[var] = t[var].astype(str)
            fig = px.bar(t, x=var, y="n", color="Situación académica", color_discrete_map=COLOR,
                         category_orders={var: orden_cat, "Situación académica": CLASES},
                         title=var, labels={"n": "%" if modo == "Porcentaje" else "Estudiantes"})
            fig.update_layout(barmode="stack")
            fig.update_xaxes(type="category")
            cols[i % 2].plotly_chart(estilo(fig, 340), use_container_width=True)

        st.markdown("#### Relación entre variables (todos los registros codificados)")
        num = limpio[[c for c in limpio.columns if not c.startswith("distrito_")]]
        corr = num.corr().round(2)
        fig = px.imshow(corr, color_continuous_scale=["#FF6B6B", "#131C2E", "#3DDC97"], zmin=-1, zmax=1,
                        aspect="auto", title="Correlación (target: 0 = Riesgo, 1 = Moderado, 2 = Alto)")
        st.plotly_chart(estilo(fig, 600), use_container_width=True)

# ===============================================================
# TAB 2: MODELOS
# ===============================================================
with tab2:
    sel = st.multiselect("Modelos a comparar", list(MODELOS), default=list(MODELOS))
    if not sel:
        st.info("Elige al menos un modelo.")
    else:
        r = resultados[resultados["Modelo"].isin(sel)].sort_values("F1-score", ascending=False)
        mejor = r.iloc[0]
        st.success(f"Mejor modelo según F1-score: **{mejor['Modelo']}** ({mejor['F1-score']:.3f}), "
                   f"con recall de {mejor['Recall_clase_Riesgo']:.2f} en la clase Riesgo.")
        c1, c2 = st.columns([1.1, 1])
        with c1:
            m = r.melt(id_vars="Modelo", value_vars=["Accuracy", "Precision", "Recall", "F1-score", "AUC-ROC"],
                       var_name="Métrica", value_name="Valor")
            fig = px.bar(m, x="Métrica", y="Valor", color="Modelo", barmode="group", title="Métricas por modelo",
                         color_discrete_sequence=[AZUL, "#B28DFF", "#3DDC97"], range_y=[0, 1])
            st.plotly_chart(estilo(fig, 400), use_container_width=True)
        with c2:
            st.dataframe(r.set_index("Modelo").style.format("{:.3f}").highlight_max(axis=0, color="#1f4fa3"),
                         use_container_width=True)
            st.caption("Evaluado sobre 100 estudiantes de prueba (20 % del total). Con pocos casos, las "
                       "métricas varían entre ejecuciones; interprétalas como orientativas.")

        st.markdown("#### Matriz de confusión")
        cm_cols = st.columns(len(sel))
        for col, nombre in zip(cm_cols, sel):
            cm = confusion_matrix(pred["real"], pred[MODELOS[nombre]], labels=[0, 1, 2])
            fig = go.Figure(go.Heatmap(z=cm, x=CLASES, y=CLASES, text=cm, texttemplate="%{text}",
                                       colorscale=[[0, "#131C2E"], [1, AZUL]], showscale=False))
            fig.update_layout(title=nombre, xaxis_title="Predicción", yaxis_title="Real")
            fig.update_yaxes(autorange="reversed")
            col.plotly_chart(estilo(fig, 340), use_container_width=True)

        st.markdown("#### Desempeño por clase")
        filas = []
        for nombre in sel:
            rep = classification_report(pred["real"], pred[MODELOS[nombre]], target_names=CLASES,
                                        output_dict=True, zero_division=0)
            for c in CLASES:
                filas.append({"Modelo": nombre, "Clase": c, "Precision": rep[c]["precision"],
                              "Recall": rep[c]["recall"], "F1": rep[c]["f1-score"]})
        pc = pd.DataFrame(filas)
        metrica = st.radio("Métrica por clase", ["Recall", "Precision", "F1"], horizontal=True)
        fig = px.bar(pc, x="Clase", y=metrica, color="Modelo", barmode="group", range_y=[0, 1],
                     color_discrete_sequence=[AZUL, "#B28DFF", "#3DDC97"])
        st.plotly_chart(estilo(fig, 340), use_container_width=True)

        if os.path.exists("graficos/curva_entrenamiento_rna.png"):
            with st.expander("Curva de entrenamiento de la red neuronal"):
                st.image("graficos/curva_entrenamiento_rna.png")

# ===============================================================
# TAB 3: VARIABLES INFLUYENTES
# ===============================================================
with tab3:
    fuentes = {"Random Forest": fi["RandomForest"], "XGBoost": fi["XGBoost"]}
    if shap_df is not None:
        fuentes["SHAP (mejor modelo)"] = shap_df.iloc[:, 0]
    c1, c2, c3 = st.columns([2, 1, 1])
    elegidos = c1.multiselect("Fuente de importancia", list(fuentes), default=list(fuentes)[:2])
    topn = c2.slider("Cuántas variables mostrar", 5, 25, 12)
    agrupar = c3.toggle("Agrupar distritos", value=False,
                        help="Suma las 20 columnas de distrito en una sola barra. Sale alta porque junta muchas columnas, no porque un distrito pese tanto.")
    if elegidos:
        tabla = pd.concat({k: fuentes[k] for k in elegidos}, axis=1)
        tabla = tabla / tabla.sum() * 100       # % del total, comparable entre fuentes
        if agrupar:
            es_d = tabla.index.str.startswith("distrito_")
            tabla = pd.concat([tabla[~es_d], tabla[es_d].sum().to_frame("Distrito (todos)").T])
        tabla.index = [i.replace("_enc", "").replace("_", " ") for i in tabla.index]
        tabla = tabla.sort_values(elegidos[0], ascending=False).head(topn)
        long = tabla.reset_index().melt(id_vars="index", var_name="Fuente", value_name="Importancia (%)")
        fig = px.bar(long, x="Importancia (%)", y="index", color="Fuente", orientation="h", barmode="group",
                     color_discrete_sequence=[AZUL, "#B28DFF", "#3DDC97"],
                     category_orders={"index": list(tabla.index)}, title="Variables que más influyen en la predicción")
        fig.update_yaxes(title="")
        st.plotly_chart(estilo(fig, 120 + 38 * len(tabla) * max(1, len(elegidos) * 0.6)), use_container_width=True)
        st.caption("Cada fuente se expresa como % del total de su propia importancia, para poder compararlas. "
                   "Las variables académicas (promedio, % aprobado, cursos pendientes) dominan porque están muy "
                   "ligadas a la situación que se predice.")
    else:
        st.info("Elige al menos una fuente.")

# ===============================================================
# TAB 4: SIMULADOR
# ===============================================================
with tab4:
    st.markdown("Ingresa el perfil de un estudiante y el modelo Random Forest estima su nivel de riesgo.")
    if not os.path.exists("modelo_random_forest.joblib"):
        st.warning("Falta modelo_random_forest.joblib. Corre de nuevo 03_entrenamiento_modelos.py.")
    else:
        pack = joblib.load("modelo_random_forest.joblib")
        rf, columnas = pack["modelo"], pack["columnas"]
        a, b, c = st.columns(3)
        with a:
            st.markdown("**Socioeconómico**")
            ciclo = st.selectbox("Ciclo", ORDEN["Ciclo"])
            ingreso = st.selectbox("Ingreso familiar", ORDEN["Ingreso familiar"], index=2)
            edu = st.selectbox("Educación de los padres", ORDEN["Educación de los padres"], index=1)
            colegio = st.radio("Colegio", ["Publico", "Privado"], horizontal=True)
            distrito = st.selectbox("Distrito", sorted(raw["Distrito"].unique()))
            acceso = st.slider("Acceso a computadora e internet", 1, 5, 3)
        with b:
            st.markdown("**Trabajo y admisión**")
            trabaja = st.radio("¿Trabaja?", ["No", "Sí"], horizontal=True)
            horas = st.selectbox("Horas de trabajo", ORDEN["Horas de trabajo"][1:], disabled=(trabaja == "No"))
            adm = st.selectbox("Percentil de admisión", ORDEN["Percentil de admisión"], index=1)
            ppa = st.selectbox("Promedio ponderado", ORDEN["Promedio ponderado"], index=2)
        with c:
            st.markdown("**Avance académico**")
            pend = st.multiselect("Ciclos con cursos pendientes",
                                  ["Del ciclo inmediato anterior", "De 2 ciclos atrás", "De 3 ciclos a más"],
                                  placeholder="Ninguno")
            npend = st.selectbox("Total de cursos pendientes", ORDEN["Cursos pendientes"])
            rend = st.slider("Rendimiento en cursos críticos", 1, 5, 3)
            pct = st.selectbox("% de cursos aprobados", ORDEN["% de cursos aprobados"], index=2)

        fila = pd.Series(0, index=columnas, dtype=float)
        fila["ciclo_enc"] = ORDEN["Ciclo"].index(ciclo)
        fila["ingreso_enc"] = ORDEN["Ingreso familiar"].index(ingreso)
        fila["acceso_tec"] = acceso
        fila["nivel_edu_padres_enc"] = ORDEN["Educación de los padres"].index(edu)
        fila["colegio_privado"] = int(colegio == "Privado")
        fila["trabaja_enc"] = int(trabaja == "Sí")
        fila["horas_trabajo_enc"] = 0 if trabaja == "No" else ORDEN["Horas de trabajo"].index(horas)
        fila["ppa_enc"] = ORDEN["Promedio ponderado"].index(ppa)
        fila["pend_ciclo_anterior"] = int("Del ciclo inmediato anterior" in pend)
        fila["pend_2_ciclos"] = int("De 2 ciclos atrás" in pend)
        fila["pend_3_o_mas_ciclos"] = int("De 3 ciclos a más" in pend)
        fila["ciclos_atraso_enc"] = 3 if fila["pend_3_o_mas_ciclos"] else 2 if fila["pend_2_ciclos"] else int(fila["pend_ciclo_anterior"])
        fila["num_cursos_pendientes_enc"] = ORDEN["Cursos pendientes"].index(npend)
        fila["rendimiento_cursos_criticos"] = rend
        fila["pct_cursos_aprobados_enc"] = ORDEN["% de cursos aprobados"].index(pct)
        fila["puntaje_admision_enc"] = 1 if adm == "No recuerdo" else ORDEN["Percentil de admisión"].index(adm)
        fila["no_recuerda_admision"] = int(adm == "No recuerdo")
        if f"distrito_{distrito}" in fila.index:
            fila[f"distrito_{distrito}"] = 1

        proba = rf.predict_proba(fila.to_frame().T[columnas])[0]
        clase = CLASES[int(np.argmax(proba))]
        r1, r2 = st.columns([1, 1.5])
        with r1:
            st.markdown(f'<div class="resultado"><div class="l" style="color:{APAGADO}">Nivel estimado</div>'
                        f'<div class="v" style="color:{COLOR[clase]}">{clase}</div>'
                        f'<div style="color:{APAGADO}">Confianza: {proba.max()*100:.0f}%</div></div>',
                        unsafe_allow_html=True)
        with r2:
            fig = go.Figure(go.Bar(x=proba * 100, y=CLASES, orientation="h",
                                   marker_color=[COLOR[c] for c in CLASES],
                                   text=[f"{p*100:.0f}%" for p in proba], textposition="auto"))
            fig.update_layout(title="Probabilidad por nivel", xaxis_title="%", xaxis_range=[0, 100])
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(estilo(fig, 230), use_container_width=True)
        st.caption("Resultado orientativo de un prototipo académico. No reemplaza la evaluación de tutoría o bienestar.")

# ===============================================================
# TAB 5: CASOS Y REPORTES
# ===============================================================
with tab5:
    st.markdown("#### Casos del conjunto de prueba")
    d_cols = [c for c in pred.columns if c.startswith("distrito_")]
    ver = pred[["real", "pred_rf", "pred_xgb", "pred_rna"]].copy()
    ver.columns = ["Real", "Random Forest", "XGBoost", "Red Neuronal (Keras)"]
    for c in ver.columns:
        ver[c] = ver[c].map(MAPA_TARGET)
    ver.insert(0, "Distrito", pred[d_cols].idxmax(axis=1).str.replace("distrito_", "", regex=False))
    ver.insert(0, "Caso", range(len(ver)))
    f1, f2 = st.columns(2)
    f_real = f1.multiselect("Situación real", CLASES, default=CLASES)
    f_modelo = f2.selectbox("Modelo para marcar aciertos", list(MODELOS))
    ver["¿Acierta?"] = np.where(ver["Real"] == ver[f_modelo], "Sí", "No")
    ver = ver[ver["Real"].isin(f_real)]
    solo_fallos = st.toggle("Mostrar solo los casos donde el modelo falla")
    if solo_fallos:
        ver = ver[ver["¿Acierta?"] == "No"]
    st.dataframe(ver, use_container_width=True, hide_index=True, height=360)
    st.caption(f"{len(ver)} casos. {(ver['¿Acierta?'] == 'Sí').mean() * 100 if len(ver) else 0:.0f}% de aciertos en esta selección.")

    st.markdown("#### Descargas")
    d1, d2, d3 = st.columns(3)
    d1.download_button("Datos filtrados (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                       "sipiera_datos_filtrados.csv", "text/csv", use_container_width=True)
    d2.download_button("Comparación de modelos (CSV)", resultados.to_csv(index=False).encode("utf-8-sig"),
                       "sipiera_comparacion_modelos.csv", "text/csv", use_container_width=True)
    d3.download_button("Predicciones de prueba (CSV)", ver.to_csv(index=False).encode("utf-8-sig"),
                       "sipiera_predicciones.csv", "text/csv", use_container_width=True)
    st.caption("Reportes pensados para que las autoridades de la FISI revisen los resultados fuera del dashboard (OE4).")

st.markdown(f'<div class="footer">SIPIERA-FISI, prototipo de tesis. Autor y tesista: <b>{AUTOR}</b>. '
            f'Universidad Nacional Mayor de San Marcos.</div>', unsafe_allow_html=True)
