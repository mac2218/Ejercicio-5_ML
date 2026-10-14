"""
Ejercicio 5 — Pipeline de scikit-learn con MLflow.

Usa tres funcionalidades de MLflow:
  1. Experiments:     cada modelo candidato es un run con parámetros, métricas y artefactos.
  2. Models:          el pipeline completo se guarda como MLflow Model con firma.
  3. Model Registry:  el mejor modelo se registra con el alias "champion" y se carga para predecir.

Uso:
  # Local (guarda todo en mlflow.db y ./mlruns; ver resultados con `mlflow ui --backend-store-uri sqlite:///mlflow.db`)
  python train_pipeline.py

  # En Databricks (requiere DATABRICKS_HOST y DATABRICKS_TOKEN, o `databricks auth login`)
  python train_pipeline.py --target databricks --experiment /Users/<tu_correo>/ejercicio5_mlflow_sklearn
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ETIQUETAS = {0: "maligno", 1: "benigno"}


# --------------------------------------------------------------------------- #
# Configuración de MLflow
# --------------------------------------------------------------------------- #
def configurar_mlflow(target: str, experimento: str) -> str:
    """Define dónde se guardan los experimentos y el registro de modelos.
    Devuelve el nombre con el que se registrará el modelo."""
    if target == "databricks":
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")  # Unity Catalog
        nombre_modelo = "workspace.default.ejercicio5_cancer_clf"
    else:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_registry_uri("sqlite:///mlflow.db")
        nombre_modelo = "ejercicio5_cancer_clf"

    mlflow.set_experiment(experimento)
    print(f"Tracking: {mlflow.get_tracking_uri()} | Experimento: {experimento}")
    return nombre_modelo


# --------------------------------------------------------------------------- #
# Datos y pipeline
# --------------------------------------------------------------------------- #
def cargar_datos(test_size: float = 0.2, seed: int = 42):
    datos = load_breast_cancer(as_frame=True)
    return train_test_split(
        datos.data, datos.target, test_size=test_size, stratify=datos.target, random_state=seed
    )


def crear_pipeline(modelo) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("modelo", modelo),
        ]
    )


def candidatos(seed: int = 42) -> dict:
    return {
        "logistic_regression": LogisticRegression(C=1.0, max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=seed),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1, random_state=seed
        ),
    }


def evaluar(pipe, X, y) -> dict:
    pred = pipe.predict(X)
    prob = pipe.predict_proba(X)[:, 1]
    return {
        "test_accuracy": accuracy_score(y, pred),
        "test_precision": precision_score(y, pred),
        "test_recall": recall_score(y, pred),
        "test_f1": f1_score(y, pred),
        "test_roc_auc": roc_auc_score(y, prob),
    }


# --------------------------------------------------------------------------- #
# Experiments + Models
# --------------------------------------------------------------------------- #
def entrenar_y_registrar(X_train, X_test, y_train, y_test) -> tuple[str, pd.DataFrame]:
    resultados = []

    with mlflow.start_run(run_name="comparacion_modelos") as run_padre:
        mlflow.set_tags({"proyecto": "ejercicio5", "dataset": "breast_cancer", "origen": "script"})
        dataset = mlflow.data.from_pandas(
            X_train.assign(target=y_train), targets="target", name="breast_cancer_train"
        )
        mlflow.log_input(dataset, context="training")

        for nombre, modelo in candidatos().items():
            with mlflow.start_run(run_name=nombre, nested=True) as run:
                pipe = crear_pipeline(modelo)

                cv_f1 = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1")
                pipe.fit(X_train, y_train)
                metricas = evaluar(pipe, X_test, y_test)
                metricas.update({"cv_f1_mean": cv_f1.mean(), "cv_f1_std": cv_f1.std()})

                mlflow.log_param("algoritmo", nombre)
                mlflow.log_params(
                    {
                        k: v
                        for k, v in modelo.get_params().items()
                        if isinstance(v, (int, float, str, bool)) or v is None
                    }
                )
                mlflow.log_metrics(metricas)

                fig, ax = plt.subplots(figsize=(4, 4))
                ConfusionMatrixDisplay.from_estimator(
                    pipe, X_test, y_test, display_labels=list(ETIQUETAS.values()), ax=ax
                )
                ax.set_title(nombre)
                mlflow.log_figure(fig, "confusion_matrix.png")
                plt.close(fig)

                info = mlflow.sklearn.log_model(
                    sk_model=pipe,
                    name="modelo",
                    signature=infer_signature(X_train, pipe.predict(X_train)),
                    input_example=X_train.head(3),
                    serialization_format="cloudpickle",
                )
                resultados.append(
                    {"modelo": nombre, "run_id": run.info.run_id, "model_uri": info.model_uri, **metricas}
                )
                print(f"  {nombre:<20} F1={metricas['test_f1']:.4f}  AUC={metricas['test_roc_auc']:.4f}")

    tabla = pd.DataFrame(resultados).sort_values(["test_f1", "test_roc_auc"], ascending=False)
    return run_padre.info.run_id, tabla


# --------------------------------------------------------------------------- #
# Model Registry
# --------------------------------------------------------------------------- #
def registrar_mejor(id_padre: str, tabla: pd.DataFrame, nombre_modelo: str) -> None:
    runs = mlflow.search_runs(
        filter_string=f"tags.mlflow.parentRunId = '{id_padre}'",
        order_by=["metrics.test_f1 DESC", "metrics.test_roc_auc DESC"],
    )
    mejor = runs.iloc[0]
    uri = tabla.set_index("run_id").loc[mejor["run_id"], "model_uri"]
    print(f"\nMejor modelo: {mejor['params.algoritmo']} (F1={mejor['metrics.test_f1']:.4f})")

    version = mlflow.register_model(model_uri=uri, name=nombre_modelo)
    cliente = MlflowClient()
    cliente.set_registered_model_alias(nombre_modelo, "champion", version.version)
    cliente.update_model_version(
        nombre_modelo,
        version.version,
        description=f"{mejor['params.algoritmo']} | test F1 = {mejor['metrics.test_f1']:.4f}",
    )
    print(f"Registrado {nombre_modelo} v{version.version} con alias 'champion'")


def predecir_con_champion(nombre_modelo: str, X, y, n: int = 10) -> pd.DataFrame:
    modelo = mlflow.pyfunc.load_model(f"models:/{nombre_modelo}@champion")
    muestra = X.head(n)
    return pd.DataFrame(
        {
            "real": y.head(n).map(ETIQUETAS).values,
            "prediccion": pd.Series(modelo.predict(muestra)).map(ETIQUETAS).values,
        }
    )


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de scikit-learn con MLflow")
    parser.add_argument("--target", choices=["local", "databricks"], default="local")
    parser.add_argument("--experiment", default=None, help="Nombre o ruta del experimento")
    args = parser.parse_args()

    experimento = args.experiment or (
        "ejercicio5_mlflow_sklearn" if args.target == "local" else "/Shared/ejercicio5_mlflow_sklearn"
    )
    nombre_modelo = configurar_mlflow(args.target, experimento)

    X_train, X_test, y_train, y_test = cargar_datos()
    print(f"Datos: {X_train.shape[0]} entrenamiento, {X_test.shape[0]} prueba\n")

    id_padre, tabla = entrenar_y_registrar(X_train, X_test, y_train, y_test)
    registrar_mejor(id_padre, tabla, nombre_modelo)

    print("\nPredicciones con el modelo 'champion':")
    print(predecir_con_champion(nombre_modelo, X_test, y_test).to_string())


if __name__ == "__main__":
    main()
