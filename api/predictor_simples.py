"""
predictor_simples.py
Carrega o modelo reduzido (6 features) para a interface simplificada.
"""
import json, os, joblib
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(_BASE, "..", "model", "rf_reduzido.joblib"))
scaler = joblib.load(os.path.join(_BASE, "..", "model", "scaler_reduzido.joblib"))
with open(os.path.join(_BASE, "..", "model", "features_reduzidas.json")) as f:
    feature_cols = json.load(f)

# Nomes amigaveis para exibicao
NOMES = {
    'sensor_11': 'Sensor Termico 1',
    'sensor_04': 'Sensor de Pressao 1',
    'sensor_09': 'Sensor Termico 2',
    'sensor_12': 'Sensor de Vibracao',
    'sensor_14': 'Sensor de Pressao 2',
    'op_setting_2': 'Modo Operacional'
}

def predict(features: dict) -> dict:
    try:
        valores = np.array([[features[col] for col in feature_cols]])
        X_s = scaler.transform(valores)
        proba = model.predict_proba(X_s)[0, 1]
        classe = int(model.predict(X_s)[0])

        return {
            'classe': classe,
            'probabilidade_falha': round(float(proba), 4),
            'status': 'falha' if classe == 1 else 'normal',
            'mensagem': (
                'Motor com FALHA IMINENTE! Realizar manutencao o quanto antes.'
                if classe == 1
                else 'Motor operando em condicoes normais.'
            )
        }
    except Exception as e:
        return {'status': 'erro', 'mensagem': str(e)}
