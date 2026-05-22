import json
import glob

notebooks = glob.glob('notebooks/*.ipynb')

# Variables that we want to replace with standard pandas read_csv to make notebooks independent of Kedro's kernel
replacements = {
    'validation_report = catalog.load("validation_report")': 'import pandas as pd; validation_report = pd.read_csv("../data/08_reporting/validation_report.csv")',
    'clf = catalog.load("model_input_clasificacion")': 'import pandas as pd; clf = pd.read_csv("../data/05_model_input/model_input_clasificacion.csv")',
    'reg = catalog.load("model_input_regresion")': 'import pandas as pd; reg = pd.read_csv("../data/05_model_input/model_input_regresion.csv")',
    'rank_clf = catalog.load("ranking_modelos_clasificacion")': 'import pandas as pd; rank_clf = pd.read_csv("../data/07_model_output/ranking_modelos_clasificacion.csv")',
    'rank_reg = catalog.load("ranking_modelos_regresion")': 'import pandas as pd; rank_reg = pd.read_csv("../data/07_model_output/ranking_modelos_regresion.csv")',
    'metricas_final = catalog.load("metricas_modelo_final")': 'import pandas as pd; metricas_final = pd.read_csv("../data/07_model_output/metricas_modelo_final.csv")',
    'optimizacion = catalog.load("comparacion_optimizacion")': 'import pandas as pd; optimizacion = pd.read_csv("../data/07_model_output/comparacion_optimizacion.csv")'
}

for nb_file in notebooks:
    with open(nb_file, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    for cell in nb.get('cells', []):
        if cell['cell_type'] != 'code':
            continue
        
        # Go through each line of the cell
        new_source = []
        for line in cell['source']:
            new_line = line
            for old_str, new_str in replacements.items():
                if old_str in line:
                    new_line = line.replace(old_str, new_str)
                    modified = True
            new_source.append(new_line)
            
        cell['source'] = new_source
                
    if modified:
        with open(nb_file, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
            
print("Notebooks have been patched with bulletproof pandas code!")
