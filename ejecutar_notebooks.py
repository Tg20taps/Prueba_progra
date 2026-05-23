"""
ejecutar_notebooks.py
=====================
Ejecuta todos los notebooks en orden, limpiando outputs anteriores.
Usar en el otro PC después de clonar el repo.

Uso:
    pip install jupyter nbconvert
    python ejecutar_notebooks.py
"""
import subprocess
import sys
from pathlib import Path

NOTEBOOKS = [
    "notebooks/01_eda_exploratorio.ipynb",
    "notebooks/02_supervised_modeling.ipynb",
    "notebooks/03_model_evaluation.ipynb",
    "notebooks/04_hyperparameter_optimization.ipynb",
    "notebooks/05_unsupervised_learning.ipynb",
    "notebooks/06_final_analysis.ipynb",
]

ROOT = Path(__file__).parent

def run_notebook(nb_path):
    full_path = ROOT / nb_path
    print(f"\n{'='*60}")
    print(f"Ejecutando: {nb_path}")
    print('='*60)
    result = subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=300",
            "--ExecutePreprocessor.kernel_name=python3",
            str(full_path)
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT)
    )
    if result.returncode == 0:
        print(f"  ✅ OK: {nb_path}")
    else:
        print(f"  ❌ ERROR en {nb_path}:")
        print(result.stderr[-1500:] if result.stderr else "Sin detalle")
    return result.returncode == 0

if __name__ == "__main__":
    print("Ejecutando todos los notebooks en orden...\n")
    resultados = []
    for nb in NOTEBOOKS:
        ok = run_notebook(nb)
        resultados.append((nb, ok))

    print("\n" + "="*60)
    print("RESUMEN FINAL")
    print("="*60)
    for nb, ok in resultados:
        estado = "✅ OK" if ok else "❌ FALLÓ"
        print(f"  {estado} — {nb}")

    fallos = [nb for nb, ok in resultados if not ok]
    if fallos:
        print(f"\n⚠️  {len(fallos)} notebook(s) fallaron. Ábrelos en Jupyter y ejecuta las celdas manualmente.")
        sys.exit(1)
    else:
        print("\n🎉 Todos los notebooks ejecutados correctamente.")
        print("Ahora puedes abrir Jupyter Lab con: jupyter lab")
