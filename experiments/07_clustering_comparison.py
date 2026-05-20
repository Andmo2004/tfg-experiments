import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime
import logging
import warnings

# Suprimir advertencias menores para salida limpia en consola
warnings.filterwarnings('ignore')

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.settings import DATASETS_CONFIG, DATASETS_DIR, RESULTS_DIR

from miclustering.data.midata import MIData # pyrefly: ignore [missing-import]
from miclustering.models.midbscan import MIDBSCAN # pyrefly: ignore [missing-import]
from miclustering.models.mikmeans import MIKMeans # pyrefly: ignore [missing-import]
from miclustering.models.mikmedoids import MIKMedoids # pyrefly: ignore [missing-import]
from miclustering.evaluation.bcm import MILEvaluator # pyrefly: ignore [missing-import]
from miclustering.distances.matrix_cache import global_persistent_cache # pyrefly: ignore [missing-import]

logging.basicConfig(level=logging.WARNING)

from miclustering.distances import DISTANCE_REGISTRY

def run_statistical_tests(df, metric_col):
    """
    Realiza Test de Friedman para ver si hay diferencias significativas globales entre modelos
    y luego Test de Wilcoxon post-hoc para comparar el mejor contra el resto.
    """
    print(f"\n{'='*50}")
    print(f" Análisis Estadístico para {metric_col}")
    print(f"{'='*50}")
    
    # Pivotar: Filas=Datasets, Columnas=Modelos, Valores=Métrica
    pivot = df.pivot(index='Dataset', columns='Model', values=metric_col).dropna()
    
    if pivot.empty or pivot.shape[1] < 2:
        print("Datos insuficientes para test estadístico (faltan métricas en algunos datasets).")
        return
        
    models = pivot.columns.tolist()
    data_matrix = [pivot[m].values for m in models]
    
    # Test de Friedman
    stat, p = friedmanchisquare(*data_matrix)
    print(f"Test de Friedman: estadístico={stat:.4f}, p-value={p:.5f}")
    
    if p < 0.05:
        print(">> Diferencias significativas detectadas (p < 0.05). Aplicando Wilcoxon post-hoc...")
        means = pivot.mean()
        best_model = means.idxmax()
        print(f"\nMejor modelo promedio: {best_model} ({means[best_model]:.4f})")
        print("\nComparaciones Pair-wise (Wilcoxon):")
        
        for m in models:
            if m != best_model:
                try:
                    stat_w, p_wilc = wilcoxon(pivot[best_model], pivot[m])  # type: ignore
                    p_wilc = float(p_wilc)  # type: ignore
                    signif = "*" if p_wilc < 0.05 else " "
                    print(f"  {best_model} vs {m:<15} : p-value = {p_wilc:.5f} {signif}")
                except Exception as e:
                    print(f"  {best_model} vs {m:<15} : Error en Wilcoxon ({e})")
    else:
        print(">> No se encontraron diferencias significativas globales entre los modelos.")

def main():
    print("Iniciando Fase 4: Comparativa de Modelos (MIDBSCAN vs MIKMeans vs MIKMedoids)...")
    
    results = []
    
    for config in DATASETS_CONFIG:
        dataset_name = config["dataset_name"]
        arff_name = config["arff_name"]
        scaler_cls = config["best_scaler"]
        metric_name = config["best_distance"]
        eps_abs = config["best_eps"]
        min_pts = config["best_min_pts"]
        
        path = os.path.join(DATASETS_DIR, f"{arff_name}.arff")
        if not os.path.exists(path):
            print(f"Saltando {dataset_name}, archivo no encontrado.")
            continue
            
        print(f"\n[+] Procesando Dataset: {dataset_name} (Métrica: {metric_name})")
        dataset = ArffToMIData.from_arff(path)
        
        # Partición 70/30 para evaluación externa justa
        train_data, test_data = dataset.split_data(percentage_train=70, seed=42)
        
        # Escalar
        scaler = scaler_cls()
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)
        
        y_true_train = np.array([int(float(bag.label)) for bag in train_scaled.bags])
        y_true_test = np.array([int(float(bag.label)) for bag in test_scaled.bags])
        
        # k_real = cantidad de clases únicas
        k_real = len(np.unique(np.concatenate([y_true_train, y_true_test])))
        if k_real < 2: k_real = 2
            
        metric_func = DISTANCE_REGISTRY.get(metric_name)
        
        # Cargar matriz de distancias (Train) para eficiencia en MIDBSCAN y MIKMedoids
        dist_matrix_train = global_persistent_cache.get(
            dataset_name=dataset_name,
            split="train",
            scaler_name=scaler_cls.__name__,
            metric_name=metric_name,
            bags=train_scaled.bags,
            metric_func=metric_func
        )
        
        models = {
            "MI-DBSCAN": MIDBSCAN(epsilon=eps_abs, min_pts=min_pts, metric=metric_name),
            "MI-KMeans": MIKMeans(k=k_real, metric=metric_name, random_state=42),
            "MI-KMedoids": MIKMedoids(k=k_real, metric=metric_name, random_state=42)
        }
        
        # Inyectar matriz precalculada
        models["MI-DBSCAN"]._distance_matrix = dist_matrix_train
        models["MI-KMedoids"]._distance_matrix = dist_matrix_train
        
        for model_name, model in models.items():
            try:
                # Entrenamiento
                model.fit(train_scaled)
                
                # Predicción en test
                test_pred_dict = model.predict(test_scaled)
                y_pred_raw_test = np.array([test_pred_dict.get(bag.bag_id, -1) for bag in test_scaled.bags])
                
                # Mapeo Húngaro
                train_pred_dict = getattr(model, "labels", {})
                if not train_pred_dict:
                    train_pred_dict = model.predict(train_scaled)

                noise_label = getattr(model, "NOISE_LABEL", -1)
                y_pred_train_raw = np.array([train_pred_dict.get(bag.bag_id, noise_label) for bag in train_scaled.bags])
                
                _, mapping = MILEvaluator.hungarian_map_clusters_to_labels(y_true_train, y_pred_train_raw)
                y_pred_mapped = np.zeros_like(y_pred_raw_test)
                for i, c in enumerate(y_pred_raw_test):
                    if c in mapping:
                        y_pred_mapped[i] = mapping[c]
                    else:
                        logging.warning(f"Clúster '{c}' encontrado en Test no estaba en el mapeo de Train. Usando fallback 0.")
                        y_pred_mapped[i] = 0
                
                # Métricas
                acc = accuracy_score(y_true_test, y_pred_mapped)
                f1 = f1_score(y_true_test, y_pred_mapped, average='weighted')
                
                results.append({
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "Accuracy": acc,
                    "F1_Score": f1
                })
                print(f"  - {model_name:<12} | F1-Score: {f1:.4f} | Accuracy: {acc:.4f}")
            except Exception as e:
                print(f"  - {model_name:<12} | ERROR: {e}")

    if not results:
        print("\n[!] No se generaron resultados validos.")
        return

    df = pd.DataFrame(results)
    
    out_dir = os.path.join(RESULTS_DIR, "fase4_comparativa_modelos")
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Guardar resultados crudos
    csv_path = os.path.join(out_dir, "resultados_modelos.csv")
    df.to_csv(csv_path, index=False)
    
    # 2. Tablas Resumen (Medias y Desviaciones)
    print("\n" + "="*50)
    print(" RESUMEN GLOBAL (Media ± Desviación Estándar)")
    print("="*50)
    
    summary = df.groupby('Model')[['Accuracy', 'F1_Score']].agg(['mean', 'std'])
    print(summary)
    summary.to_csv(os.path.join(out_dir, "resumen_global.csv"))
    
    # 3. Tests Estadísticos
    run_statistical_tests(df, "Accuracy")
    run_statistical_tests(df, "F1_Score")
    
    # 4. Gráficas Comparativas
    print("\nGenerando gráficas comparativas...")
    
    # Boxplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.boxplot(data=df, x='Model', y='F1_Score', ax=axes[0], palette="Set2")
    axes[0].set_title('Comparativa F1-Score')
    axes[0].set_ylabel('F1-Score (Weighted)')
    
    sns.boxplot(data=df, x='Model', y='Accuracy', ax=axes[1], palette="Set3")
    axes[1].set_title('Comparativa Accuracy')
    axes[1].set_ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparativa_boxplots.png"))
    plt.show()
    plt.close()
    
    # Barplots (Media por Modelo con intervalo de confianza)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.barplot(data=df, x='Model', y='F1_Score', ax=axes[0], palette="Set2", capsize=.1)
    axes[0].set_title('F1-Score Promedio')
    axes[0].set_ylabel('F1-Score Medio')
    
    sns.barplot(data=df, x='Model', y='Accuracy', ax=axes[1], palette="Set3", capsize=.1)
    axes[1].set_title('Accuracy Promedio')
    axes[1].set_ylabel('Accuracy Media')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparativa_barplots.png"))
    plt.show()
    plt.close()
    
    # Gráfica de calor Dataset vs Modelo (F1)
    pivot_f1 = df.pivot(index='Dataset', columns='Model', values='F1_Score')
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_f1, annot=True, cmap="YlGnBu", fmt=".3f", cbar_kws={'label': 'F1-Score'})
    plt.title('Rendimiento (F1-Score) por Dataset y Modelo')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "heatmap_f1_score.png"))
    plt.show()
    plt.close()

    print(f"\n[+] Proceso completado. Gráficas y tablas guardadas en {out_dir}")

if __name__ == "__main__":
    main()
