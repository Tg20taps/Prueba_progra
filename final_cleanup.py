import json
import glob

notebooks = glob.glob('notebooks/*.ipynb')

for nb_file in notebooks:
    with open(nb_file, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    for cell in nb.get('cells', []):
        if cell['cell_type'] != 'code':
            continue
        
        # Remove any line that I accidentally injected containing 'catalog.load'
        new_source = [line for line in cell['source'] if 'catalog.load' not in line]
        
        # Also remove any lines I might have injected with the pandas replacements just in case
        new_source = [line for line in new_source if 'import pandas as pd;' not in line]
        
        if len(new_source) != len(cell['source']):
            cell['source'] = new_source
            modified = True
                
    if modified:
        with open(nb_file, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
            
print("Cleanup complete!")
