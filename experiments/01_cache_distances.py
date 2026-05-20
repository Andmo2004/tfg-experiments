from miclustering.distances import DISTANCE_REGISTRY
"""
00_distance_matrix_cache.py

Script para precomputar las matrices de distancia y almacenarlas en la caché persistente (disco).
Esto permite ahorrar muchísimo tiempo en las fases siguientes del experimento.
"""

import os
import sys
from miclustering.data.arff_reader import ArffToMIData
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.settings import DATASETS_DIR, RESULTS_DIR, DATASETS_CONFIG
from miclustering.data.midata import MIData # pyrefly: ignore [missing-import]
from miclustering.preprocessing.scaler import MinMaxScaler, StandardScaler # pyrefly: ignore [missing-import]
from miclustering.distances.matrix_cache import global_persistent_cache # pyrefly: ignore [missing-import]

logging.basicConfig(level=logging.INFO, format="%(message)s")

SCALERS = {
    "MinMaxScaler": MinMaxScaler,
    "StandardScaler": StandardScaler
}

def precompute_matrices():
    print("=" * 80)
    print("PRECOMPUTANDO MATRICES DE DISTANCIA (CACHÉ PERSISTENTE)")
    print("=" * 80)

    output_dir = os.path.join(RESULTS_DIR, "distance_matrices")
    os.makedirs(output_dir, exist_ok=True)

    # Seed
    seed = 42
    
    for config in DATASETS_CONFIG:
        dataset_name = config["dataset_name"]
        arff_name = config["arff_name"]
        
        print(f"\n► Procesando Dataset: {dataset_name}")
        path = os.path.join(DATASETS_DIR, f"{arff_name}.arff")
        if not os.path.exists(path):
            print(f"  [!] No se encontró el archivo: {path}. Omitiendo.")
            continue
            
        dataset_full = ArffToMIData.from_arff(path)
        train_data, _ = dataset_full.split_data(percentage_train=70, seed=seed)
        
        available_metrics = list(DISTANCE_REGISTRY.keys())
        if dataset_name in ["Harddrive1", "Thioredoxin", "Newsgroups1"]:
            if "earth_movers" in available_metrics:
                available_metrics.remove("earth_movers")
        
        for scaler_name, ScalerClass in SCALERS.items():
            scaler = ScalerClass()
            scaled_train = scaler.fit_transform(train_data)
            
            for metric_name in available_metrics:
                metric_func = DISTANCE_REGISTRY[metric_name]
                
                # Precompute for train split
                global_persistent_cache.get(
                    dataset_name=dataset_name,
                    split="train",
                    scaler_name=scaler_name,
                    metric_name=metric_name,
                    bags=scaled_train.bags,
                    metric_func=metric_func,
                    seed=seed
                )
                
        # Liberar memoria de la caché local para el siguiente dataset (ya está guardado en disco)
        global_persistent_cache.clear_memory()

    print("\n" + "=" * 80)
    print("PRECOMPUTACIÓN COMPLETADA")
    print("=" * 80)

if __name__ == "__main__":
    precompute_matrices()
