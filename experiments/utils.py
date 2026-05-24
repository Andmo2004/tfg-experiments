"""
experiments/utils.py

Utilidades compartidas por los scripts de experimentos del TFG.
Actualmente expone cleanup_phase(), una función de limpieza de memoria
que debe llamarse al final de cada fase para evitar OOM en Kaggle/notebooks.
"""

import gc
import logging

logger = logging.getLogger(__name__)


def cleanup_phase(*vars_to_delete) -> int:
    """Libera variables grandes y fuerza el recolector de basura de CPython.

    Debe llamarse al final de cada script de fase (o al final de cada
    iteración de dataset en bucles largos) para evitar que las matrices
    NxN y los modelos entrenados sigan ocupando RAM en el proceso siguiente.

    Pasos que realiza:
      1. Elimina las referencias a los objetos pasados como argumentos.
      2. Vacía la caché en memoria de PersistentDistanceMatrixCache
         (las entradas en disco se conservan intactas).
      3. Llama a gc.collect() para forzar la liberación inmediata.

    Args:
        *vars_to_delete: Variables a eliminar antes del GC.
                         Pasa aquí dist_matrix, model, train_scaled, etc.

    Returns:
        Número de objetos recogidos por el GC (útil para debugging).

    Ejemplo de uso al final de un script de fase:
        from utils import cleanup_phase
        cleanup_phase(results, studies)

    Ejemplo de uso dentro de un bucle de datasets:
        for config in DATASETS_CONFIG:
            dist_matrix = ...
            model = ...
            # ... lógica de la fase ...
            cleanup_phase(dist_matrix, model)
    """
    # Importación local para no forzar el import en todos los módulos
    # que importen utils sin necesitar la caché
    try:
        from miclustering.distances.matrix_cache import global_persistent_cache
        global_persistent_cache.clear_memory()
    except ImportError:
        logger.warning("[cleanup_phase] No se pudo importar global_persistent_cache.")

    for var in vars_to_delete:
        del var

    collected = gc.collect()
    logger.info("Objetos liberados por GC: {collected}")
    print("Objetos liberados por GC: {collected}")
    return collected