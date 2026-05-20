import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.settings import DATASETS_CONFIG, DATASETS_DIR, RESULTS_DIR
from miclustering.data.midata import MIData
from miclustering.data.bag import Bag
from miclustering.data.instance import Instance
from miclustering.preprocessing.scaler import MinMaxScaler
from miclustering.models.midbscan import MIDBSCAN
from miclustering.evaluation.bcm import MILEvaluator
from miclustering.distances.hausdorff import hausdorff_distance, hausdorff_distance_avg
from miclustering.distances.probability_distribution import mahalanobis_distance
from miclustering.distances.distance_matrix import compute_distance_matrix
from miclustering.distances.matrix_cache import global_persistent_cache

logging.basicConfig(level=logging.WARNING)

METRICS_TO_TEST = {
    "hausdorff": hausdorff_distance,
    "hausdorff_avg": hausdorff_distance_avg,
    "mahalanobis": mahalanobis_distance
}

def inject_noise_into_dataset(dataset: MIData, noise_ratio: float = 0.10, noise_magnitude: float = 10.0) -> MIData:
    """
    Inyecta una instancia de ruido extremo en el `noise_ratio` (ej. 10%) de las bolsas.
    El dataset asume que ya está normalizado [0, 1], por lo que un valor de 10.0 es un outlier claro.
    """
    noisy_bags = []
    random.seed(42)
    
    for bag in dataset.bags:
        # Clonamos la bolsa y sus instancias
        new_instances = [Instance(list(inst.values), inst.schema) for inst in bag.instances]
        
        if random.random() < noise_ratio:
            # Crear una instancia sintética con valores extremos
            schema = bag.instances[0].schema
            num_features = len(schema)
            extreme_values = [noise_magnitude] * num_features
            noisy_instance = Instance(extreme_values, schema)
            new_instances.append(noisy_instance)
            
        noisy_bags.append(Bag(bag.bag_id, bag.label, new_instances))
        
    return MIData(noisy_bags, dataset.name + "_noisy")

def main():
    print("Iniciando Prueba de Robustez al Ruido...")
    results = []
    
    for config in DATASETS_CONFIG:
        dataset_name = config["dataset_name"]
        path = os.path.join(DATASETS_DIR, f"{config['arff_name']}.arff")
        if not os.path.exists(path): continue
            
        print(f"\n[+] Procesando: {dataset_name}")
        dataset = MIData.from_arff(path)
        
        # 1. Normalizar el dataset limpio
        scaler = MinMaxScaler()
        clean_scaled = scaler.fit_transform(dataset)
        y_true = np.array([int(float(bag.label)) for bag in clean_scaled.bags])
        
        # 2. Generar el dataset ruidoso (10% de bolsas afectadas)
        noisy_scaled = inject_noise_into_dataset(clean_scaled, noise_ratio=0.10, noise_magnitude=10.0)
        
        for metric_name, metric_func in METRICS_TO_TEST.items():
            print(f"  - Evaluando {metric_name}...")

            scaler_name = "MinMaxScaler"
            dist_clean = global_persistent_cache.get(
                dataset_name=dataset_name,
                split="train",          
                scaler_name=scaler_name,
                metric_name=metric_name,
                bags=clean_scaled.bags,
                metric_func=metric_func
            )

            dist_noisy = global_persistent_cache.get(
                dataset_name=f"{dataset_name}_noisy",
                split="full",
                scaler_name="MinMaxScaler",
                metric_name=metric_name,
                bags=noisy_scaled.bags,
                metric_func=metric_func,
                save=False,
            )
            
            best_eps = config["best_eps"]
            
            # --- EVALUACIÓN LIMPIA ---
            model_clean = MIDBSCAN(epsilon=best_eps, min_pts=config["best_min_pts"], metric=metric_name)
            model_clean._distance_matrix = dist_clean
            model_clean.fit(clean_scaled)
            
            pred_clean = np.array([model_clean.labels.get(b.bag_id, -1) for b in clean_scaled.bags])
            _, map_clean = MILEvaluator.hungarian_map_clusters_to_labels(y_true, pred_clean)
            f1_clean = f1_score(y_true, [map_clean.get(c, 0) for c in pred_clean], average='weighted')
            
            # --- EVALUACIÓN CON RUIDO ---
            model_noisy = MIDBSCAN(epsilon=best_eps, min_pts=config["best_min_pts"], metric=metric_name)
            model_noisy._distance_matrix = dist_noisy
            model_noisy.fit(noisy_scaled)
            
            pred_noisy = np.array([model_noisy.labels.get(b.bag_id, -1) for b in noisy_scaled.bags])
            _, map_noisy = MILEvaluator.hungarian_map_clusters_to_labels(y_true, pred_noisy)
            f1_noisy = f1_score(y_true, [map_noisy.get(c, 0) for c in pred_noisy], average='weighted')
            
            drop_pct = ((f1_clean - f1_noisy) / f1_clean) * 100 if f1_clean > 0 else 0
            
            results.append({
                "Dataset": dataset_name,
                "Metric": metric_name,
                "F1_Clean": f1_clean,
                "F1_Noisy": f1_noisy,
                "Drop_Percentage": drop_pct
            })

            global_persistent_cache.clear_memory()

    df = pd.DataFrame(results)
    out_dir = os.path.join(RESULTS_DIR, "fase4_robustez")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "resultados_robustez.csv"), index=False)
    
    # Gráfico de caída de rendimiento
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Dataset", y="Drop_Percentage", hue="Metric", palette="Set1")
    plt.title("Caída del F1-Score tras inyectar ruido (10% bolsas)", fontsize=14)
    plt.ylabel("% de Caída en Rendimiento", fontsize=12)
    plt.xlabel("Dataset", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "barplot_robustez_caida.png"), dpi=300)
    plt.close()
    
    print(f"\n[+] Resultados de robustez guardados en {out_dir}")

if __name__ == "__main__":
    main()