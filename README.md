# Proyecto Santander — Santander CX: Análisis de Propensión al Cambio Bancario

Proyecto académico de análisis de datos sobre **propensión al cambio de banco ("churn")** en
clientes mexicanos, a partir de una encuesta de preferencias (300 respuestas, 21 preguntas). El
repositorio reúne dos entregables complementarios:

| Componente | Descripción | Ubicación |
|---|---|---|
| 🌐 **Sitio / presentación** | Página web con la narrativa, hallazgos y visualizaciones del proyecto. | [`index.html`](index.html) · [ver publicado](https://leoroldanp.github.io/Proyecto-Santander/) |
| 🤖 **Pipeline de ML** | Notebooks reproducibles: definición del target, modelado, evaluación y segmentación. | [`ML_Pipeline/`](ML_Pipeline/) |

---

## 🤖 Pipeline de Machine Learning

Pipeline limpio y auditable para estimar el riesgo de fuga de cada cliente. La documentación
completa (metodología, resultados y limitaciones) está en **[`ML_Pipeline/README.md`](ML_Pipeline/README.md)**.

- **`pipeline_churn.ipynb`** — Fase 1: pipeline base end-to-end (target NPS+Kano, anti-leakage,
  comparación de 3 modelos, evaluación con SHAP).
- **`pipeline_churn_v2.ipynb`** — Fase 1.5: interacciones de features, optimización del umbral de
  decisión y segmentación de clientes con K-Means.

> [!IMPORTANT]
> La etiqueta objetivo `Y` es **sintética**: se construye a partir de las respuestas (score NPS +
> Kano), no de un churn observado. El modelo reproduce una _definición_ de riesgo; el valor está en
> la metodología, no en una predicción validada contra deserción real. La Fase 2 requeriría churn
> real observado.

### Inicio rápido

```bash
cd ML_Pipeline
pip install -r requirements.txt
jupyter notebook        # ejecutar las celdas de arriba hacia abajo
```

### Resultados clave

- **Modelo ganador:** Regresión Logística — ROC-AUC holdout ≈ 0.66, F1 ≈ 0.61 (con umbral optimizado).
- **Segmentación (K-Means, K=3):** 3 arquetipos; el cluster "Híbrido exigente" concentra el mayor
  riesgo de fuga (74 %).

---

## 🛠️ Stack

**Datos / ML:** Python · pandas · scikit-learn · XGBoost · SHAP · matplotlib · seaborn
**Web:** HTML · CSS · Chart.js

> La encuesta es **anónima** (solo marca temporal y respuestas, sin datos personales).
