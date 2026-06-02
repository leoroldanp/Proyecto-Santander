# ML Pipeline — Propensión al cambio bancario (Santander CX)

Pipeline de Machine Learning para estimar la **propensión de un cliente a cambiar de banco**
("churn") a partir de una encuesta de preferencias de clientes bancarios en México (300
respuestas, 21 preguntas). El énfasis está en una **metodología transparente y auditable**: cada
decisión queda justificada en el notebook.

> [!IMPORTANT]
> **La etiqueta objetivo `Y` es _sintética_.** No existe dato real de quién se cambió de banco, así
> que `Y` ("cliente en riesgo de fuga") se construye a partir de las propias respuestas mediante un
> score de riesgo justificado (NPS + Kano). **El modelo aprende a reproducir una _definición_ de
> riesgo, no un churn observado.** El valor del trabajo está en la metodología de scoring, no en una
> capacidad predictiva probada contra deserción real. Pasar a churn real es la [Fase 2](#-fase-2--lo-que-falta).

---

## 📂 Estructura

```
ML_Pipeline/
├── README.md                       # este archivo
├── requirements.txt                # dependencias (versiones fijadas)
├── data/
│   └── encuesta_preferencia_entidades_financieras.csv   # encuesta anónima (300×21)
├── pipeline_churn.ipynb            # Fase 1   — pipeline base end-to-end
└── pipeline_churn_v2.ipynb         # Fase 1.5 — interacciones + umbral óptimo + segmentación K-Means
```

## 🚀 Cómo ejecutar

```bash
# 1. (recomendado) entorno virtual
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate

# 2. dependencias
pip install -r requirements.txt

# 3. abrir los notebooks
jupyter notebook        # o jupyter lab / VS Code
```

Ejecuta las celdas de arriba hacia abajo. La ruta al CSV (`DATA_PATH`) se resuelve sola tanto si
abres el notebook desde la raíz del repo como desde `ML_Pipeline/`. Todo es determinista
(`random_state=42`).

> Requiere Python 3.10+ (probado en 3.12, Anaconda).

---

## 🧠 Metodología

### 1. Definición del target `Y` — score NPS + Kano

`Y` se construye con un **score compuesto continuo y un único umbral** (no un OR de binarios):

```
risk_score = TRUST_POINTS[trust_level] + IF_Kano
Y = 1  si  risk_score >= 2.75   (si no, 0)
```

- **Confianza graduada estilo NPS** — la escala de confianza 1-5 se mapea a las bandas NPS
  (detractor / pasivo / promotor). Los detractores profundos (1-2) pesan fuerte (`2.5`), el
  detractor débil (3) pesa parcial (`1.0`), el pasivo (4) es neutral (`0.0`) y el promotor (5)
  recibe un **descuento de lealtad** (`-1.0`) que reduce falsos positivos en la base más valiosa.
- **Índice de Fuga (`IF`) ponderado por Modelo Kano** — cada razón de cambio marcada por el cliente
  pesa según su gravedad: Must-Be `2.0` · Performance financiero `1.5` · Performance operativo
  `1.25` · Delighter `0.8` · Influencia externa `0.5` · Lealtad `0.0`.
- **Umbral 2.75** — calibrado para un balance **50/50** (verificado: `P(Y=1) = 0.500`) y para exigir
  "más de una señal" antes de marcar riesgo. Se valida un **gradiente NPS** claro: detractor
  profundo/sólido 100 % → débil 73 % → pasivo 44 % → promotor 28 % en `Y=1`.

### 2. Prevención de _leakage_

Se **excluyen como predictores todas las variables que definen `Y`** (`trust_level`,
`switch_reasons` y derivados, `risk_score`, etc.). De lo contrario el modelo "haría trampa"
reconstruyendo la fórmula en vez de aprender señales independientes. Quedan **16 features base**:
conteos de multi-selects, escalas operativas, ordinales (`age`, `income`) y categóricas
(`main_bank`, `inst_type`, …).

### 3. Preprocesamiento y modelado

- `ColumnTransformer`: mediana + escalado (numéricas), moda + `OrdinalEncoder` (ordinales),
  constante + `OneHotEncoder` (categóricas).
- Split estratificado **80/20** (`random_state=42`).
- Tres modelos en pipelines idénticos: **Regresión Logística**, **Random Forest**, **XGBoost**, con
  manejo de balance de clases. Comparación por `StratifiedKFold(5)`; ganador por ROC-AUC; tuning con
  `GridSearchCV`.
- Evaluación en el holdout: matriz de confusión, curvas ROC / PR / calibración, mini-auditoría de
  _fairness_ por subgrupos e interpretabilidad con **SHAP**.

### 4. Fase 1.5 (`v2`) — mejoras quirúrgicas

- **5 interacciones** de features con hipótesis de negocio explícita, filtradas por señal
  (correlación de Spearman vs `Y`; se descarta `|ρ| < 0.05`).
- **Optimización del umbral de decisión** maximizando F1 sobre probabilidades _cross-validated_ del
  train (nunca sobre el holdout — eso sería leakage).
- **Segmentación no supervisada (K-Means)** que descubre arquetipos de cliente con features
  disjuntas de `Y`, y los cruza con la tasa de riesgo para priorizar retención.

---

## 📊 Resultados

### Modelo (holdout, n = 60)

| Métrica | Baseline v1 (umbral 0.50) | v2 interacciones (0.50) | v2 interacciones + umbral óptimo (0.44) |
|---|---|---|---|
| ROC-AUC | 0.650 | 0.661 | 0.661 |
| F1 | 0.588 | 0.560 | **0.613** |
| Precisión | 0.714 | 0.700 | 0.594 |
| Recall | 0.500 | 0.467 | **0.633** |

**Modelo ganador: Regresión Logística** (`C=0.01`). El _lift_ en F1/recall proviene sobre todo de
**optimizar el umbral**, no de las interacciones (que en un holdout de 60 personas tienen efecto
marginal). ROC-AUC apenas se mueve: honesto y esperable.

### Segmentación (K-Means, K=3)

| Cluster | Arquetipo | n | Riesgo `Y=1` |
|---|---|---|---|
| 1 | **Híbrido exigente en riesgo** — usa banca + fintech, joven, hiper-involucrado | 73 | **74 %** |
| 0 | Joven mainstream de bajo uso | 117 | 44 % |
| 2 | Adulto consolidado fluido (ingreso alto, multi-app, baja fricción) | 110 | 40 % |

El **Cluster 1** es el objetivo prioritario de retención. _Nota honesta:_ las silhouettes son bajas
(~0.13) → la estructura de clusters es débil; los arquetipos son orientativos, no fronteras nítidas.

---

## 🔭 Fase 2 — lo que falta

El techo de desempeño **no está en el modelo, está en la data**: 300 encuestas y una `Y` sintética.
Para una predicción con valor real, la Fase 2 necesita **churn observado**:

1. Un **identificador** que enlace cada respuesta con un cliente real.
2. Una **ventana de tiempo** (p. ej. 6-12 meses).
3. El **resultado real** (`churn = 1/0`) desde los sistemas del banco.

Con eso, `Y` deja de ser una fórmula y pasa a ser un hecho medido — y el modelo adquiere validez
empírica, no solo metodológica.

---

## ⚙️ Stack

`pandas` · `numpy` · `scikit-learn` · `xgboost` · `shap` · `matplotlib` · `seaborn` (ver
`requirements.txt`).

> Proyecto académico de análisis de datos. La encuesta es **anónima** (solo marca temporal y
> respuestas; sin datos personales).
