# SIPIERA-FISI

Desarrollo y evaluación de un modelo de Machine Learning para la predicción del éxito académico basado en variables socioeconómicas y académicas de estudiantes de la FISI (UNMSM).

Tesis de pregrado — Bryan Samuel Sulca López. Asesor: Daniel Alfonso Quinto Pazce. Desarrollo de Proyecto de Tesis II, 2026-II.

## Qué hace

Estima el nivel de éxito académico de un estudiante de IX o X ciclo (En riesgo / Regular / Buen rendimiento) a partir de un formulario de 15 ítems. Compara Random Forest, XGBoost y una red neuronal, y los muestra en un dashboard de Streamlit con cinco pestañas (Panorama, Comparación de modelos, Variables influyentes, Simulador de riesgo, Casos y reportes).

## Datos

- 69 respuestas reales del formulario y 431 registros sintéticos (columna `origen_dato` = REAL / SINTETICO).
- Los modelos se evalúan **solo con los 69 registros reales**, con validación cruzada estratificada 5×2 y SMOTE-NC aplicado únicamente al entrenamiento de cada pliegue.
- Las respuestas reales **no se publican** en este repositorio para proteger la identidad de los estudiantes. Para reproducir el pipeline, el archivo `dataset_sipiera_500_real_y_sintetico.xlsx` se coloca en la misma carpeta que los scripts (disponible para el asesor bajo solicitud).

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
python 02_limpieza_y_encoding.py        # genera dataset_limpio.csv
python 03_entrenamiento_modelos.py      # genera resultados_metricas.csv, modelos y gráficos
streamlit run 05_dashboard_streamlit.py # abre el dashboard en el navegador
```

Semilla fija: 42.

## Archivos

| Archivo | Descripción |
|---|---|
| `02_limpieza_y_encoding.py` | Lee el Excel, codifica las variables y genera `dataset_limpio.csv` |
| `03_entrenamiento_modelos.py` | Entrena y evalúa los modelos; genera métricas, importancia de variables y SHAP |
| `05_dashboard_streamlit.py` | Dashboard del prototipo |
| `resultados_metricas.csv` | Métricas de la validación cruzada (media y desviación) |
| `*.png` | Figuras de resultados |

## Limitaciones conocidas

- Solo 69 registros reales: los resultados son orientativos y no permiten declarar un modelo mejor que otro.
- La variable objetivo es autorreportada y se solapa con variables académicas (PPA, % de cursos aprobados, cursos pendientes); el modelo sin ellas está en desarrollo.
- SMOTE-NC todavía trata algunas variables ordinales como continuas (corrección pendiente).
- La red neuronal no generaliza con este tamaño de muestra.
