# SIPIERA-FISI

Desarrollo y evaluación de un modelo de Machine Learning para la predicción del éxito académico basado en variables socioeconómicas y académicas de estudiantes de la FISI (UNMSM).

Tesis de pregrado — Bryan Samuel Sulca López. Asesor: Daniel Alfonso Quinto Pazce. Desarrollo de Proyecto de Tesis II, 2026-II.

## Qué hace

Estima el nivel de éxito académico de un estudiante de IX o X ciclo (En riesgo / Regular / Buen rendimiento) a partir de un formulario de 15 ítems. Compara Random Forest, XGBoost y una red neuronal, y los muestra en un dashboard de Streamlit con cinco pestañas (Panorama, Comparación de modelos, Variables influyentes, Simulador de riesgo, Casos y reportes).

## Datos

- 69 respuestas reales del formulario y 431 registros sintéticos (columna `origen_dato` = REAL / SINTETICO). Los sintéticos se generaron con SMOTE (imbalanced-learn 0.14.2, k_neighbors = 5, random_state = 42) sobre las variables ordinales; las variables nominales y los bloques lógicos (p. ej. trabaja + horas) se copiaron del vecino real más cercano de la misma clase, descartando duplicados.
- SMOTE-NC (algoritmo de `imbalanced-learn`) se usa solo dentro del entrenamiento de cada pliegue de la validación cruzada.
- Los modelos se evalúan **solo con los 69 registros reales**, con validación cruzada estratificada 5×2 y SMOTE-NC aplicado únicamente al entrenamiento de cada pliegue.
- Las respuestas reales **no se publican** en este repositorio para proteger la identidad de los estudiantes. Para reproducir el pipeline, el archivo `dataset_sipiera_500_real_y_sintetico.xlsx` se coloca en la misma carpeta que los scripts (disponible para el asesor bajo solicitud).

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
python 02_limpieza_y_encoding.py        # genera dataset_limpio.csv
python 03_entrenamiento_modelos.py      # genera resultados_metricas.csv, modelos y gráficos
python 04_validacion_pi13.py            # validación PI13: M1/M2, líneas base, pruebas estadísticas (carpeta evidencias/)
streamlit run 05_dashboard_streamlit.py # abre el dashboard en el navegador
```

Semilla fija: 42.

## Archivos

| Archivo | Descripción |
|---|---|
| `02_limpieza_y_encoding.py` | Lee el Excel, codifica las variables y genera `dataset_limpio.csv` |
| `03_entrenamiento_modelos.py` | Entrena y evalúa los modelos; genera métricas, importancia de variables y SHAP |
| `04_validacion_pi13.py` | Validación cruzada 5×10 solo con registros reales; modelo M1 (sin variables académicas solapadas) y M2 (referencia); líneas base; t corregida con Holm; importancia por permutación |
| `evidencias/` | Salidas de la validación: resultados por modelo y pliegue, pruebas estadísticas, importancia de variables, integridad y figuras |
| `05_dashboard_streamlit.py` | Dashboard del prototipo |
| `resultados_metricas.csv` | Métricas de la validación cruzada (media y desviación) |
| `*.png` | Figuras de resultados |

## Limitaciones conocidas

- Solo 69 registros reales: los resultados son orientativos y no permiten declarar un modelo mejor que otro.
- La variable objetivo es autorreportada y se solapa con variables académicas (PPA, % de cursos aprobados, cursos pendientes); el modelo sin ellas está en desarrollo.
- En el script 04, SMOTE-NC redondea las ordinales a niveles válidos y aplica la regla trabaja = 0 → horas = 0; el script 03 aún no tiene esa corrección.
- Con las variables socioeconómicas y de admisión (M1), ningún modelo supera de forma significativa a las líneas base; el F1 macro de Random Forest es 0,371 (ver `evidencias/`).
- La red neuronal no generaliza con este tamaño de muestra.
