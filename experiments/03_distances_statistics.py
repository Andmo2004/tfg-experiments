from miclustering.distances import DISTANCE_REGISTRY
import os
import sys
from miclustering.data.arff_reader import ArffToMIData
import time
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score, accuracy_score, f1_score
import scikit_posthocs as sp
from miclustering.distances.matrix_cache import global_persistent_cache

import logging
import warnings

warnings.filterwarnings('ignore')

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.settings import DATASETS_CONFIG, DATASETS_DIR, RESULTS_DIR

from miclustering.data.midata import MIData
from miclustering.preprocessing.scaler import MinMaxScaler
from miclustering.models.midbscan import MIDBSCAN
from miclustering.evaluation.bcm import MILEvaluator
from miclustering.distances.distance_matrix import compute_distance_matrix

logging.basicConfig(level=logging.WARNING)

def get_bag_centroids(dataset):
    centroids = []
    for bag in dataset.bags:
        centroids.append(np.mean(bag.as_matrix(), axis=0))
    return np.array(centroids)

def generate_plots(csv_path, out_dir):
    """
    Genera gráficos similares al paper: Boxplots con stripplot y Diagramas de Diferencias Críticas (Nemenyi).
    """
    if not os.path.exists(csv_path):
        print(f"No se encontró el archivo {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    os.makedirs(out_dir, exist_ok=True)
    
    metrics = ["Silhouette", "Davies_Bouldin", "Accuracy", "F1_Score"]
    
    for metric in metrics:
        if metric not in df.columns: continue
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='Metric', y=metric, palette="vlag", showfliers=False, width=0.6)
        sns.stripplot(data=df, x='Metric', y=metric, color=".2", alpha=0.6, jitter=True, size=5)
        
        plt.title(f'Distribución de {metric} por Distancia', fontsize=14)
        plt.xlabel('Medida de Distancia', fontsize=12)
        plt.ylabel(f'Índice {metric}', fontsize=12)
        plt.xticks(rotation=15)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plot_path = os.path.join(out_dir, f"boxplot_{metric}.png")
        plt.savefig(plot_path, dpi=300)
        plt.show()
        plt.close()
        print(f"Guardado boxplot para {metric} en {plot_path}")

    for metric in metrics:
        if metric not in df.columns: continue
        
        pivot_df = df.pivot(index='Dataset', columns='Metric', values=metric).dropna()
        if pivot_df.empty or pivot_df.shape[1] < 2: continue
        
        if sp is None:
            print("scikit_posthocs no disponible, se omiten los diagramas Nemenyi.")
            continue
        
        ascending = True if metric == "Davies_Bouldin" else False
        
        nemenyi_results = sp.posthoc_nemenyi_friedman(pivot_df.values)
        nemenyi_results.columns = pivot_df.columns
        nemenyi_results.index = pivot_df.columns
        
        plt.figure(figsize=(10, 4))
        plt.title(f'Test de Nemenyi para {metric} (a=0.05)', pad=20)
        
        try:
            sp.sign_plot(nemenyi_results, alpha=0.05)
        except AttributeError:
            print("Versión de scikit-posthocs no soporta diagrama CD directo, usando heatmap.")
            sns.heatmap(nemenyi_results < 0.05, annot=True, cmap="Blues", cbar=False)
            
        plot_path = os.path.join(out_dir, f"nemenyi_{metric}.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()
        print(f"Guardado test Nemenyi para {metric} en {plot_path}")

def main():
    print("Iniciando Fase 3: Estudio del Impacto de la Distancia (DBSCAN-MIL)...")
    results = []
    
    for config in DATASETS_CONFIG:
        dataset_name = config["dataset_name"]
        arff_name = config["arff_name"]
        
        path = os.path.join(DATASETS_DIR, f"{arff_name}.arff")
        if not os.path.exists(path):
            continue
            
        print(f"\n[+] Procesando Dataset: {dataset_name}")
        dataset = ArffToMIData.from_arff(path)
        
        scaler = MinMaxScaler()
        scaled_dataset = scaler.fit_transform(dataset)
        
        y_true = np.array([int(float(bag.label)) for bag in scaled_dataset.bags])
        X_centroids = get_bag_centroids(scaled_dataset)
        
        for metric_name, metric_func in DISTANCE_REGISTRY.items():
            print(f"  - Métrica evaluada: {metric_name}")
            
            model = MIDBSCAN(epsilon=0.368, min_pts=2, metric=metric_name)
            
            start_time = time.time()
            
            dist_matrix = global_persistent_cache.get(
                dataset_name=dataset_name,
                split="full",
                scaler_name="MinMaxScaler",
                metric_name=metric_name,
                bags=scaled_dataset.bags,
                metric_func=metric_func,
                save=True,   
            )
            model._distance_matrix = dist_matrix
            model.fit(scaled_dataset)
            exec_time = time.time() - start_time

            pred_dict = getattr(model, "labels", {})
            y_pred_raw = np.array([pred_dict.get(bag.bag_id, -1) for bag in scaled_dataset.bags])
            
            if len(np.unique(y_pred_raw)) < 2:
                continue
                
            try:
                dist_matrix_copy = dist_matrix.copy()
                np.fill_diagonal(dist_matrix_copy, 0)
                sil = silhouette_score(dist_matrix_copy, y_pred_raw, metric="precomputed")
            except Exception:
                sil = np.nan
                
            try:
                db = davies_bouldin_score(X_centroids, y_pred_raw)
            except:
                db = np.nan
                
            _, mapping = MILEvaluator.hungarian_map_clusters_to_labels(y_true, y_pred_raw)
            y_pred_mapped = np.array([mapping.get(c, 0) for c in y_pred_raw])
            
            acc = accuracy_score(y_true, y_pred_mapped)
            f1 = f1_score(y_true, y_pred_mapped, average='weighted')
            
            results.append({
                "Dataset": dataset_name,
                "Metric": metric_name,
                "Silhouette": sil,
                "Davies_Bouldin": db,
                "Accuracy": acc,
                "F1_Score": f1,
                "Exec_Time_Secs": exec_time
            })

    if not results:
        print("\n[!] No se generaron resultados validos.")
        return

    df = pd.DataFrame(results).dropna()
    out_dir = os.path.join(RESULTS_DIR, "fase3_distancias")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "resultados_dbscan_tiempos.csv")
    df.to_csv(csv_path, index=False)

    # Gráfico Trade-off (Tiempo vs F1-Score)
    print(f"\n{'='*50}")
    print(" Generando gráfico de Trade-off (Time vs F1-Score)")
    print(f"{'='*50}")
    
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df, 
        x="Exec_Time_Secs", 
        y="F1_Score", 
        hue="Metric", 
        style="Dataset", 
        s=150, 
        alpha=0.8,
        palette="Set2"
    )
    plt.title("Trade-off: Complejidad Temporal vs Rendimiento (MIDBSCAN)", fontsize=14)
    plt.xlabel("Tiempo de Ejecución (Segundos)", fontsize=12)
    plt.ylabel("F1-Score", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tradeoff_time_vs_f1.png"), dpi=300)
    plt.show()
    plt.close()
    
    print(f"[+] Gráfico Trade-off guardado en {out_dir}")
    
    # Generar gráficos estilo paper
    print(f"\n{'='*50}")
    print(" Generando gráficos estilo paper (Boxplots y Nemenyi)")
    print(f"{'='*50}")
    plots_dir = os.path.join(out_dir, "plots")
    generate_plots(csv_path, plots_dir)

if __name__ == "__main__":
    main()