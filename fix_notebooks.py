import os
import json
import glob

notebooks = glob.glob('notebooks/*.ipynb')

# Variables that often get forgotten to be loaded in the user's notebooks
missing_vars = {
    'validation_report': 'validation_report = catalog.load("validation_report")\n',
    'clf': 'clf = catalog.load("model_input_clasificacion")\n',
    'reg': 'reg = catalog.load("model_input_regresion")\n',
    'rank_clf': 'rank_clf = catalog.load("ranking_modelos_clasificacion")\n',
    'rank_reg': 'rank_reg = catalog.load("ranking_modelos_regresion")\n',
    'metricas_final': 'metricas_final = catalog.load("metricas_modelo_final")\n',
    'optimizacion': 'optimizacion = catalog.load("comparacion_optimizacion")\n'
}

for nb_file in notebooks:
    with open(nb_file, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    for cell in nb.get('cells', []):
        if cell['cell_type'] != 'code':
            continue
        
        source = cell['source']
        if not source:
            continue
            
        source_str = ''.join(source)
        
        for var, load_str in missing_vars.items():
            # If variable is used in the cell but not loaded via catalog.load or defined
            # We specifically check common patterns to avoid injecting unnecessarily
            if var in source_str and 'catalog.load' not in source_str and f'{var} =' not in source_str:
                cell['source'].insert(0, load_str)
                modified = True
                print(f"Fixed missing '{var}' in {nb_file}")
                
            # Special case for rank_clf and rank_reg if they are being copied like rank_clf_viz = rank_clf.copy()
            elif f"{var}.copy()" in source_str and 'catalog.load' not in source_str:
                 cell['source'].insert(0, load_str)
                 modified = True
                 print(f"Fixed missing '{var}' (.copy()) in {nb_file}")
                 
            elif f"clean_text_columns({var})" in source_str and 'catalog.load' not in source_str:
                 cell['source'].insert(0, load_str)
                 modified = True
                 print(f"Fixed missing '{var}' (clean_text) in {nb_file}")

            elif f"(\"clasificacion\", {var})" in source_str and 'catalog.load' not in source_str:
                 cell['source'].insert(0, load_str)
                 modified = True
                 print(f"Fixed missing '{var}' in {nb_file}")

            elif f"(\"regresion\", {var})" in source_str and 'catalog.load' not in source_str:
                 cell['source'].insert(0, load_str)
                 modified = True
                 print(f"Fixed missing '{var}' in {nb_file}")
                 
    if modified:
        with open(nb_file, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
            
print("All notebooks fixed successfully!")
