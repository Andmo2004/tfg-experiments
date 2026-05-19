import os
import sys
import json
import argparse
import logging
from datetime import datetime
import numpy as np
from sklearn import metrics

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from miclustering.data.midata import MIData
from miclustering.preprocessing.scaler import MinMaxScaler, StandardScaler
from miclustering.models.midbscan import MIDBSCAN
from miclustering.models.miknn import MIKnn
from miclustering.models.mikmeans import MIKMeans
from miclustering.models.mikmedoids import MIKMedoids
from miclustering.evaluation.bcm import MILEvaluator

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from miclustering.distances.matrix_cache import global_persistent_cache
from miclustering.distances.hausdorff import hausdorff_distance, hausdorff_distance_min, hausdorff_distance_avg
from miclustering.distances.probability_distribution import cauchy_schwarz_distance, earth_movers_distance, mahalanobis_distance

DISTANCES_REGISTRY = {
    "hausdorff": hausdorff_distance,
    "hausdorff_min": hausdorff_distance_min,
    "hausdorff_avg": hausdorff_distance_avg,
    "cauchy_schwarz": cauchy_schwarz_distance,
    "earth_movers": earth_movers_distance,
    "mahalanobis": mahalanobis_distance
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def evaluate_predictions(y_true, y_pred):
    """Evalúa las predicciones frente a las etiquetas reales."""
    acc = metrics.accuracy_score(y_true, y_pred)
    prec = metrics.precision_score(y_true, y_pred, zero_division=0, average='weighted')
    rec = metrics.recall_score(y_true, y_pred, zero_division=0, average='weighted')
    f1 = metrics.f1_score(y_true, y_pred, zero_division=0, average='weighted')
    return {
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1-Score": float(f1)
    }

def normalize_key(key: str) -> str:
    """Normaliza las claves para aceptar variaciones en el nombre (ej. con o sin espacios)."""
    return key.lower().replace(" ", "_")

def main():
    parser = argparse.ArgumentParser(description="Punto de entrada CLI para ejecutar experimentos de MIL.")
    parser.add_argument("--config", type=str, required=True, help="Ruta al archivo JSON de configuración.")
    parser.add_argument("--output_dir", type=str, default="results", help="Directorio para guardar resultados.")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        logger.error(f"Archivo de configuración no encontrado: {args.config}")
        return

    # Leer JSON
    with open(args.config, "r", encoding="utf-8") as f:
        raw_config = json.load(f)
        
    config = {normalize_key(k): v for k, v in raw_config.items()}

    dataset_name = config.get("dataset", "musk1")
    distance_metric = config.get("medida_de_distancia", config.get("medida_distancia", "hausdorff"))
    scaler_name = config.get("metodo_de_escalado", config.get("metodo_escalado", "MinMaxScaler"))
    seed = config.get("semilla", 42)
    eval_metric = config.get("metrica_de_rendimiento_a_optimizar", config.get("metrica_optimizacion", "F1-Score"))
    algorithm = config.get("algoritmo", "midbscan").lower()
    hyperparams = config.get("hiperparametros", {})
    
    optimizar_optuna = config.get("optimizar_optuna", False)
    optuna_trials = config.get("optuna_trials", 30)

    logger.info("=" * 60)
    logger.info(f"Ejecutando configuración:")
    logger.info(f"  Dataset: {dataset_name}")
    logger.info(f"  Algoritmo: {algorithm}")
    logger.info(f"  Distancia: {distance_metric}")
    logger.info(f"  Semilla: {seed}")
    logger.info(f"  Optimizar (Optuna): {optimizar_optuna}")
    logger.info("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    dataset_path = os.path.join(current_dir, "datasets", f"{dataset_name}.arff")
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset no encontrado en la ruta esperada: {dataset_path}")
        return

    dataset = MIData.from_arff(dataset_path)
    train_data, test_data = dataset.split_data(percentage_train=70, seed=seed)
    
    logger.info(f"Datos cargados: Train ({train_data.get_num_bags()} bolsas), Test ({test_data.get_num_bags()} bolsas)")

    # Ejecutar el preprocesado (Escalado)
    if scaler_name == "StandardScaler":
        scaler = StandardScaler()
    elif scaler_name == "MinMaxScaler":
        scaler = MinMaxScaler()
    else:
        scaler = None

    if scaler:
        logger.info(f"Aplicando escalado: {scaler_name}")
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)
    else:
        logger.info("No se aplica escalado.")
        train_scaled = train_data
        test_scaled = test_data

    y_true_train = np.array([int(float(bag.label)) for bag in train_scaled.bags])
    y_true_test = np.array([int(float(bag.label)) for bag in test_scaled.bags])

    # Instanciar el modelo
    model_classes = {
        "midbscan": MIDBSCAN,
        "miknn": MIKnn,
        "mikmeans": MIKMeans,
        "mikmedoids": MIKMedoids
    }

    if algorithm not in model_classes:
        logger.error(f"Algoritmo desconocido: {algorithm}. Disponibles: {list(model_classes.keys())}")
        return

    ModelClass = model_classes[algorithm]
    metric_func = DISTANCES_REGISTRY.get(str(distance_metric), hausdorff_distance)  # type: ignore
    
    # Optimización de Hiperparámetros (Optuna)
    dist_matrix = None
    if algorithm in ["midbscan", "mikmedoids"]:
        logger.info("Precalculando/cargando matriz de distancias para eficiencia...")
        dist_matrix = global_persistent_cache.get(
            dataset_name=str(dataset_name),
            split="train",
            scaler_name=str(scaler_name) if scaler_name else "none",
            metric_name=str(distance_metric),
            bags=train_scaled.bags,
            metric_func=metric_func
        )

    if optimizar_optuna:
        logger.info(f"Iniciando optimización con Optuna ({optuna_trials} trials)...")
        
        def objective(trial):
            model = None
            if algorithm == "midbscan":
                min_pts = trial.suggest_int("min_pts", 2, 20)
                eps_percentile = trial.suggest_float("eps_percentile", 1.0, 40.0)
                
                upper = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]  # type: ignore
                upper_positive = upper[upper > 0]
                if len(upper_positive) == 0:
                    raise optuna.exceptions.TrialPruned()
                eps_absolute = float(np.percentile(upper_positive, eps_percentile))
                
                model = MIDBSCAN(epsilon=eps_absolute, min_pts=min_pts, metric=str(distance_metric))
                model._distance_matrix = dist_matrix
                
            elif algorithm == "miknn":
                k = trial.suggest_int("k", 1, 15)
                model = MIKnn(k=k, metric=str(distance_metric))
                
            elif algorithm == "mikmeans":
                k = trial.suggest_int("k", 2, 15)
                model = MIKMeans(k=k, metric=str(distance_metric), random_state=seed)
                
            elif algorithm == "mikmedoids":
                k = trial.suggest_int("k", 2, 15)
                model = MIKMedoids(k=k, metric=str(distance_metric), random_state=seed)
                model._distance_matrix = dist_matrix  # type: ignore
                
            if model is None:
                raise ValueError("Modelo no inicializado.")

            try:
                # Evaluación
                if algorithm == "miknn":
                    train_sub, val_sub = train_scaled.split_data(percentage_train=80, seed=seed)
                    model.fit(train_sub)
                    preds = model.predict(val_sub)
                    y_pred_val = np.array([preds.get(bag.bag_id, 0) for bag in val_sub.bags])
                    y_true_val = np.array([int(float(bag.label)) for bag in val_sub.bags])
                    return float(metrics.f1_score(y_true_val, y_pred_val, zero_division=0, average='weighted'))
                else:
                    model.fit(train_scaled)
                    if algorithm == "midbscan" and getattr(model, "cluster_count", -1) == 0:
                        return 0.0
                        
                    train_pred_dict = getattr(model, "labels", {})
                    if not train_pred_dict:
                        train_pred_dict = model.predict(train_scaled)
                    
                    noise_label = getattr(model, "NOISE_LABEL", -1)
                    y_pred_train_raw = np.array([train_pred_dict.get(bag.bag_id, noise_label) for bag in train_scaled.bags])
                    _, mapping = MILEvaluator.hungarian_map_clusters_to_labels(y_true_train, y_pred_train_raw)
                    
                    y_pred_mapped = np.zeros_like(y_pred_train_raw)
                    for i, c in enumerate(y_pred_train_raw):
                        y_pred_mapped[i] = mapping.get(c, 0)
                        
                    return float(metrics.f1_score(y_true_train, y_pred_mapped, zero_division=0, average='weighted'))
            except Exception:
                raise optuna.exceptions.TrialPruned()

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=optuna_trials)
        
        best_params = study.best_params
        logger.info(f"Mejores hiperparámetros encontrados: {best_params}")
        
        # Actualizar hyperparams
        if algorithm == "midbscan":
            hyperparams["min_pts"] = best_params["min_pts"]
            upper = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]  # type: ignore
            upper_positive = upper[upper > 0]
            if len(upper_positive) > 0:
                hyperparams["epsilon"] = float(np.percentile(upper_positive, best_params["eps_percentile"]))
            else:
                hyperparams["epsilon"] = 1.0
        else:
            hyperparams["k"] = best_params["k"]
            
        # Añadir al output para registro
        hyperparams["optuna_optimized"] = True

    # Inyectar métrica si es necesaria
    if "metric" not in hyperparams:
        hyperparams["metric"] = distance_metric
        
    # Inicialización estándar
    if algorithm in ["mikmeans", "mikmedoids"]:
        if "random_state" not in hyperparams:
            hyperparams["random_state"] = seed
            
    # Filtramos kwargs para no inyectar 'optuna_optimized' al constructor
    init_kwargs = {k: v for k, v in hyperparams.items() if k != "optuna_optimized"}
    
    logger.info(f"Instanciando modelo con hiperparámetros finales: {init_kwargs}")
    try:
        model = ModelClass(**init_kwargs)
        if algorithm in ["midbscan", "mikmedoids"] and dist_matrix is not None:
            model._distance_matrix = dist_matrix  # type: ignore
    except Exception as e:
        logger.error(f"Error al instanciar el modelo: {e}")
        return

    # Entrenar y Predecir
    logger.info("Entrenando modelo...")
    try:
        model.fit(train_scaled)
    except Exception as e:
        logger.error(f"Error en el entrenamiento (fit): {e}")
        return
        
    logger.info("Generando predicciones sobre el conjunto de test...")
    try:
        pred_dict = model.predict(test_scaled)
    except Exception as e:
        logger.error(f"Error en la predicción (predict): {e}")
        return
        
    y_pred_raw_test = np.array([pred_dict.get(bag.bag_id, -1) for bag in test_scaled.bags])

    # Mapeo de clusters a clases reales si es un modelo no supervisado
    if algorithm == "miknn":
        # Supervisado (clasificación directa)
        y_pred_test = y_pred_raw_test
    else:
        # No supervisado (clustering)
        logger.info("Modelo de clustering detectado: Mapeando clústeres a etiquetas (Hungarian Mapping)...")
        # Obtener labels de train para mapear
        if hasattr(model, "labels"):
            train_pred_dict = model.labels
        else:
            train_pred_dict = model.predict(train_scaled)
            
        noise_label = getattr(model, "NOISE_LABEL", -1)
        y_pred_train_raw = np.array([train_pred_dict.get(bag.bag_id, noise_label) for bag in train_scaled.bags])
        
        _, mapping = MILEvaluator.hungarian_map_clusters_to_labels(y_true_train, y_pred_train_raw)
        
        y_pred_test = np.zeros_like(y_pred_raw_test)
        for i, cluster in enumerate(y_pred_raw_test):
            if cluster in mapping:
                y_pred_test[i] = mapping[cluster]
            else:
                y_pred_test[i] = 0 # Valor por defecto si un cluster no apareció en train
                
    # Evaluar métricas de rendimiento
    results = evaluate_predictions(y_true_test, y_pred_test)
    
    # Añadir estadísticas del modelo si están disponibles
    if hasattr(model, "get_statistics"):
        stats = model.get_statistics()
        if "num_clusters" in stats:
            results["num_clusters"] = stats["num_clusters"]
        if "noise_percentage" in stats:
            results["noise_percentage"] = stats["noise_percentage"]

    # Guardar los resultados automáticamente
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(args.output_dir, f"run_{algorithm}_{dataset_name}_{timestamp}.json")
    
    output_data = {
        "timestamp": timestamp,
        "input_config": raw_config,
        "results": results
    }
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)
        
    logger.info(f"Ejecución finalizada con éxito. Resultados guardados en: {out_file}")
    logger.info("Resumen de métricas obtenidas:")
    for k, v in results.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"  {k}: {v}")

if __name__ == "__main__":
    main()
