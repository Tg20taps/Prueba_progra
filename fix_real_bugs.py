import json

# =============================================
# FIX NOTEBOOK 2: Missing rank_clf / rank_reg load cell
# =============================================
with open('notebooks/02_supervised_modeling.ipynb', 'r', encoding='utf-8') as f:
    nb2 = json.load(f)

for cell in nb2['cells']:
    if cell['cell_type'] == 'code' and ''.join(cell['source']).strip() == '## 2. Objetivos supervisados':
        cell['source'] = [
            'rank_clf = read_csv(DATA_OUTPUT / "ranking_modelos_clasificacion.csv")\n',
            'rank_reg = read_csv(DATA_OUTPUT / "ranking_modelos_regresion.csv")\n',
            'clf_data = read_csv(DATA_MODEL / "model_input_clasificacion.csv")\n',
            'reg_data = read_csv(DATA_MODEL / "model_input_regresion.csv")\n',
            'print(f"Clasificacion: {len(rank_clf)} modelos evaluados")\n',
            'print(f"Regresion: {len(rank_reg)} modelos evaluados")\n',
        ]
        cell['outputs'] = []
        cell['execution_count'] = None
        print("Fixed Notebook 2: empty cell now loads rank_clf and rank_reg")
        break

with open('notebooks/02_supervised_modeling.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb2, f, indent=1)


# =============================================
# FIX NOTEBOOK 3: Missing y_proba_c definition
# Insert the classification evaluation code BEFORE the ROC cell
# =============================================
with open('notebooks/03_model_evaluation.ipynb', 'r', encoding='utf-8') as f:
    nb3 = json.load(f)

roc_cell_idx = None
for i, cell in enumerate(nb3['cells']):
    if cell['cell_type'] == 'code' and 'y_proba_c is not None' in ''.join(cell['source']):
        roc_cell_idx = i
        break

if roc_cell_idx is not None:
    # Insert missing cell that defines y_proba_c before the ROC cell
    missing_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": "eval_clf_missing",
        "metadata": {},
        "outputs": [],
        "source": [
            "from sklearn.model_selection import train_test_split\n",
            "from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report\n",
            "\n",
            "FEATURES_NUMERICAS_C = [\n",
            '    "peso_kg", "volumen_m3", "distancia_km", "tiempo_estimado_hrs",\n',
            '    "peaje_total", "capacidad_kg", "capacidad_m3", "km_recorridos",\n',
            '    "eficiencia_peso", "eficiencia_volumen", "costo_por_km",\n',
            '    "id_ruta", "id_vehiculo", "mes_envio", "dia_semana_envio", "trimestre_envio",\n',
            "]\n",
            "FEATURES_CATEGORICAS_C = [\n",
            '    "estado", "tipo_carga_norm", "origen", "destino",\n',
            '    "tipo_via", "tipo", "estado_vehiculo",\n',
            "]\n",
            "\n",
            "feats_c = [c for c in FEATURES_NUMERICAS_C + FEATURES_CATEGORICAS_C if c in clf_data.columns]\n",
            'X_clf = clf_data[feats_c]\n',
            'y_clf = clf_data["tiene_incidencia"]\n',
            "\n",
            "X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(\n",
            "    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf\n",
            ")\n",
            "\n",
            "y_pred_c = modelo_clf.predict(X_test_c)\n",
            "y_proba_c = modelo_clf.predict_proba(X_test_c)[:, 1] if hasattr(modelo_clf, 'predict_proba') else None\n",
            "\n",
            "cm = confusion_matrix(y_test_c, y_pred_c)\n",
            "print(classification_report(y_test_c, y_pred_c, target_names=['Sin incidencia', 'Con incidencia']))\n",
            "\n",
            "disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Sin incidencia', 'Con incidencia'])\n",
            "disp.plot(cmap='Blues')\n",
            "import matplotlib.pyplot as plt\n",
            'plt.title("Matriz de Confusion - GaussianNB")\n',
            "plt.tight_layout()\n",
            "plt.show()\n",
        ]
    }
    nb3['cells'].insert(roc_cell_idx, missing_cell)
    print(f"Fixed Notebook 3: inserted missing classification eval cell before ROC cell (was index {roc_cell_idx})")
else:
    print("ROC cell not found in Notebook 3 - no changes made")

with open('notebooks/03_model_evaluation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb3, f, indent=1)

print("\nAll fixes applied successfully!")
