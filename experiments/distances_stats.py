import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score, accuracy_score, f1_score
from datetime import datetime
import logging
import warnings

# Suprimir advertencias menores para salida limpia en consola
warnings.filterwarnings('ignore')

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from config.settings import DATASETS_CONFIG, DATASETS_DIR, RESULTS_DIR
from miclustering.data.midata import MIData
from miclustering.preprocessing.scaler import MinMaxScaler
from miclustering.models.mikmedoids import MIKMedoids
from miclustering.evaluation.bcm import MILEvaluator
from miclustering.distances.hausdorff import hausdorff_distance, hausdorff_distance_min, hausdorff_distance_avg
from miclustering.distances.probability_distribution import cauchy_schwarz_distance, mahalanobis_distance
from miclustering.distances.matrix_cache import global_persistent_cache

logging.basicConfig(level=logging.WARNING)

# Excluimos Earth Movers Distance por su alto coste computacional en algunos datasets,
# pero incluimos las principales distancias del proyecto.
DISTANCES = {
    "hausdorff": hausdorff_distance,
    "hausdorff_min": hausdorff_distance_min,
    "hausdorff_avg": hausdorff_distance_avg,
    "cauchy_schwarz": cauchy_schwarz_distance,
    "mahalanobis": mahalanobis_distance
}

def get_bag_centroids(dataset):
    """Calcula el centroide medio de cada bolsa para métricas basadas en características (ej. Davies-Bouldin)"""
    centroids = []
    for bag in dataset.bags:
        centroids.append(np.mean(bag.as_matrix(), axis=0))
    return np.array(centroids)

def run_statistical_tests(df, metric_col, maximize=True):
    """
    Realiza Test de Friedman para ver si hay diferencias significativas globales
    y luego Test de Wilcoxon post-hoc para comparar la mejor distancia contra el resto.
    """
    print(f"\n{'='*50}")
    print(f" Análisis Estadístico para {metric_col}")
    print(f"{'='*50}")
    
    # Pivotar: Filas=Datasets, Columnas=Distancias, Valores=Métrica
    pivot = df.pivot(index='Dataset', columns='Metric', values=metric_col).dropna()
    
    if pivot.empty or pivot.shape[1] < 2:
        print("Datos insuficientes para test estadístico (faltan métricas en algunos datasets).")
        return
        
    distances = pivot.columns.tolist()
    data_matrix = [pivot[d].values for d in distances]
    
    # Test de Friedman
    stat, p = friedmanchisquare(*data_matrix)
    print(f"Test de Friedman: estadístico={stat:.4f}, p-value={p:.5f}")
    
    if p < 0.05:
        print(">> Diferencias significativas detectadas (p < 0.05). Aplicando Wilcoxon post-hoc...")
        means = pivot.mean()
        best_dist = means.idxmax() if maximize else means.idxmin()
        print(f"\nMejor distancia promedio: {best_dist} ({means[best_dist]:.4f})")
        print("\nComparaciones Pair-wise (Wilcoxon):")
        
        for d in distances:
            if d != best_dist:
                try:
                    stat_w, p_wilc = wilcoxon(pivot[best_dist], pivot[d])
                    signif = "*" if p_wilc < 0.05 else " "
                    print(f"  {best_dist} vs {d:<15} : p-value = {p_wilc:.5f} {signif}")
                except Exception as e:
                    print(f"  {best_dist} vs {d:<15} : Error en Wilcoxon ({e})")
    else:
        print(">> No se encontraron diferencias significativas globales entre las distancias.")

def main():
    print("Iniciando Fase 3: Estudio del Impacto de la Distancia...")
    
    results = []
    
    for config in DATASETS_CONFIG:
        dataset_name = config["dataset_name"]
        arff_name = config["arff_name"]
        
        path = os.path.join(DATASETS_DIR, f"{arff_name}.arff")
        if not os.path.exists(path):
            continue
            
        print(f"\n[+] Procesando Dataset: {dataset_name}")
        dataset = MIData.from_arff(path)
        
        # Estandarizamos escalar en [0,1] para este experimento de aislamiento de métrica
        scaler = MinMaxScaler()
        scaled_dataset = scaler.fit_transform(dataset)
        
        y_true = np.array([int(float(bag.label)) for bag in scaled_dataset.bags])
        k_real = len(np.unique(y_true))
        if k_real < 2: k_real = 2
            
        X_centroids = get_bag_centroids(scaled_dataset)
        
        for metric_name, metric_func in DISTANCES.items():
            print(f"  - Métrica evaluada: {metric_name}")
            
            # Cargar/Calcular la matriz desde caché persistente
            dist_matrix = global_persistent_cache.get(
                dataset_name=dataset_name,
                split="full",
                scaler_name="MinMaxScaler",
                metric_name=metric_name,
                bags=scaled_dataset.bags,
                metric_func=metric_func
            )
            
            # Utilizamos MIKMedoids en lugar de MIDBSCAN para evitar que un hyperparámetro
            # dependiente de la escala como "epsilon" contamine la comparativa pura de distancias.
            # Al darle K real, forzamos la partición y medimos la bondad de la distancia.
            model = MIKMedoids(k=k_real, metric=metric_name, random_state=42)
            model._distance_matrix = dist_matrix
            model.fit(scaled_dataset)
            
            y_pred_raw = np.array([model.labels.get(bag.bag_id, 0) for bag in scaled_dataset.bags])
            
            if len(np.unique(y_pred_raw)) < 2:
                # El modelo colapsó todos los puntos a 1 clúster
                continue
                
            # MÉTRICAS INTERNAS
            try:
                # Copia para garantizar que la diagonal sea exactamente cero
                dist_matrix_copy = dist_matrix.copy()
                np.fill_diagonal(dist_matrix_copy, 0)
                sil = silhouette_score(dist_matrix_copy, y_pred_raw, metric="precomputed")  # type: ignore
            except Exception:
                sil = np.nan
                
            try:
                # DB usa los centroides generados
                db = davies_bouldin_score(X_centroids, y_pred_raw)  # type: ignore
            except:
                db = np.nan
                
            # MÉTRICAS EXTERNAS
            # Modelos particionales requieren Hungarian mapping para emparejar etiquetas clúster con reales
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
                "F1_Score": f1
            })

    if not results:
        print("\n[!] No se generaron resultados validos.")
        return

    # Limpiamos resultados NaN
    df = pd.DataFrame(results).dropna()
    
    out_dir = os.path.join(RESULTS_DIR, "fase3_distancias")
    os.makedirs(out_dir, exist_ok=True)
    
    csv_path = os.path.join(out_dir, "resultados_distancias.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[+] Métricas crudas guardadas en {csv_path}")

    # TESTS ESTADÍSTICOS (FRIEDMAN & WILCOXON)
    run_statistical_tests(df, "Silhouette", maximize=True)
    run_statistical_tests(df, "Davies_Bouldin", maximize=False)
    run_statistical_tests(df, "F1_Score", maximize=True)

    # ESTUDIO DE CORRELACIÓN (Spearman/Pearson)
    print(f"\n{'='*50}")
    print(" Estudio de Correlación (Métrica Interna vs Externa)")
    print(f"{'='*50}")
    
    metrics_to_corr = [("Silhouette", "F1_Score"), ("Davies_Bouldin", "F1_Score")]
    
    for int_m, ext_m in metrics_to_corr:
        spearman_corr, sp_p = spearmanr(df[int_m], df[ext_m])
        pearson_corr, pe_p = pearsonr(df[int_m], df[ext_m])
        
        print(f"\nCorrelación {int_m} vs {ext_m}:")
        print(f"  Spearman: {spearman_corr:.4f} (p-value={sp_p:.4f})")
        print(f"  Pearson:  {pearson_corr:.4f} (p-value={pe_p:.4f})")
        
        # Gráfica de Dispersión
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x=int_m, y=ext_m, hue="Metric", style="Dataset", s=100)
        plt.title(f"Correlación: {int_m} vs {ext_m}\nSpearman: {spearman_corr:.2f} | Pearson: {pearson_corr:.2f}")
        
        # Añadir línea de tendencia
        sns.regplot(data=df, x=int_m, y=ext_m, scatter=False, color='gray', line_kws={"linestyle": "--", "alpha": 0.5})
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"corr_{int_m}_{ext_m}.png"))
        plt.close()
        
    print(f"\n[+] Gráficos de correlación generados en {out_dir}")

if __name__ == "__main__":
    main()
