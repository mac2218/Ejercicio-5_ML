# Ejercicio 5 — Pipeline de scikit-learn con MLflow

Pipeline de clasificación (tumor **maligno / benigno**, dataset *Breast Cancer Wisconsin* de scikit-learn) adaptado para usar MLflow en Databricks.

## Funcionalidades de MLflow utilizadas

| Funcionalidad | Uso en el código |
|---|---|
| **Experiments** | Un *run* padre `comparacion_modelos` con un *run* hijo por modelo. Se registran parámetros, métricas (validación cruzada y prueba), el dataset de entrenamiento y la matriz de confusión. |
| **Models** | El pipeline completo (imputación + escalado + modelo) se guarda con `mlflow.sklearn.log_model`, con firma y ejemplo de entrada. |
| **Model Registry** | El mejor modelo (por F1) se registra, recibe el alias `champion` y se carga con `models:/<nombre>@champion` para predecir. En Databricks se usa Unity Catalog. |

Modelos comparados: regresión logística, Random Forest y Gradient Boosting.

## Estructura

```
├── train_pipeline.py                        # Versión en archivo de Python
├── notebooks/
│   └── ejercicio5_mlflow_databricks.ipynb   # Versión en Jupyter Notebook (Databricks)
├── requirements.txt
└── README.md
```

## Cómo ejecutarlo

### Notebook en Databricks
Importar `notebooks/ejercicio5_mlflow_databricks.ipynb` en el workspace y ejecutar **Run all** con cómputo Serverless.

### Script en local
```bash
pip install -r requirements.txt
python train_pipeline.py
mlflow ui --backend-store-uri sqlite:///mlflow.db   # abrir http://127.0.0.1:5000
```

### Script contra Databricks
```bash
pip install -r requirements.txt databricks-sdk
databricks auth login --host https://<tu-workspace>.cloud.databricks.com
python train_pipeline.py --target databricks --experiment /Users/<tu_correo>/ejercicio5_mlflow_sklearn
```

## Resultados

| Modelo | F1 (prueba) | ROC AUC |
|---|---|---|
| **Regresión logística** (`champion`) | **0.986** | **0.995** |
| Gradient Boosting | 0.966 | 0.989 |
| Random Forest | 0.958 | 0.994 |
