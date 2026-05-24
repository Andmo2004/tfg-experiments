$ErrorActionPreference = "Stop"

# Activar entorno virtual
& ".\.venv\Scripts\Activate.ps1"

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar experimentos
$phases = @("01", "02", "03", "04", "05", "06", "07", "08", "09", "10")

foreach ($phase in $phases) {
    Get-ChildItem "experiments/${phase}_*.py" | ForEach-Object {
        python $_.FullName

        if ($LASTEXITCODE -ne 0) {
            throw "Error ejecutando $($_.FullName)"
        }
    }
}