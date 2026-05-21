$ErrorActionPreference = "Stop"

# Activar entorno virtual
& ".\.venv\Scripts\Activate.ps1"

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar experimentos
$phases = @("03", "04", "05")
# $phases = @("00", "01", "02", "03", "04", "05")

foreach ($phase in $phases) {
    Get-ChildItem "experiments/${phase}_*.py" | ForEach-Object {
        python $_.FullName

        if ($LASTEXITCODE -ne 0) {
            throw "Error ejecutando $($_.FullName)"
        }
    }
}