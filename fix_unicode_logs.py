import sys, os, re

# Fix all non-ASCII chars in logger calls
files_to_fix = [
    r'src\database\pipelines\data_cleaning\nodes.py',
    r'src\database\pipelines\data_ingestion\nodes.py',
    r'src\database\pipelines\data_transformation\nodes.py',
    r'src\database\pipelines\data_validation\nodes.py',
    r'src\database\pipelines\model_input\nodes.py',
    r'src\database\pipelines\model_training\nodes.py',
    r'src\database\pipelines\hyperparameter_tuning\nodes.py',
    r'src\database\pipelines\unsupervised_learning\nodes.py',
]

replacements = {
    '\u2192': '->',
    '\u274c': '[ERROR]',
    '\u2705': '[OK]',
    '\u26a0': '[WARN]',
    '\u2713': '[OK]',
    '\u2717': '[FAIL]',
    '\u2014': '--',
}

root = r'c:\Users\Tg20taps\Desktop\Correcion prueba 2\database'

for rel in files_to_fix:
    path = os.path.join(root, rel)
    try:
        text = open(path, encoding='utf-8').read()
        lines = text.split('\n')
        new_lines = []
        changed = False
        for line in lines:
            if 'logger.' in line and any(ord(c) > 127 for c in line):
                fixed = line
                for bad, good in replacements.items():
                    fixed = fixed.replace(bad, good)
                # Replace any remaining non-ASCII with safe version
                fixed_encoded = fixed.encode('ascii', errors='replace').decode('ascii')
                new_lines.append(fixed_encoded)
                if fixed_encoded != line:
                    changed = True
                    print("FIXED:", rel, "|", fixed_encoded.strip()[:80])
            else:
                new_lines.append(line)
        if changed:
            with open(path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(new_lines))
            print("Saved:", rel)
    except Exception as e:
        print("ERROR", rel, str(e))

print("Done.")
