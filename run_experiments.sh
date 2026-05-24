#!/bin/bash

set -e

# Activar entorno virtual
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar experimentos
for phase in 01 02 03 04 05 06 07 08 09; do
    for file in experiments/${phase}_*.py; do
        python "$file"
    done
done