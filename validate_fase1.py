import pandas as pd

clf = pd.read_csv("data/05_model_input/model_input_clasificacion.csv", encoding="latin-1")
reg = pd.read_csv("data/05_model_input/model_input_regresion.csv", encoding="latin-1")
rep = pd.read_csv("data/08_reporting/model_input_report.csv", encoding="latin-1")

print("=== model_input_clasificacion ===")
print(f"Shape: {clf.shape}")
print(f"Target (tiene_incidencia): {clf['tiene_incidencia'].value_counts().to_dict()}")
print(f"Total nulls: {clf.isnull().sum().sum()}")
print(f"Columns ({len(clf.columns)}): {list(clf.columns)}")

print()
print("=== model_input_regresion ===")
print(f"Shape: {reg.shape}")
print(f"Target - mean={reg['dias_en_transito'].mean():.2f}, median={reg['dias_en_transito'].median():.2f}, std={reg['dias_en_transito'].std():.2f}")
print(f"Total nulls: {reg.isnull().sum().sum()}")
print(f"Columns ({len(reg.columns)}): {list(reg.columns)}")

print()
print("=== model_input_report ===")
print(rep.to_string())
