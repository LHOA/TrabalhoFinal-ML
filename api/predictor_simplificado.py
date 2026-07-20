"""
predictor_simplificado.py
Carrega ambos os modelos (XGBoost e Random Forest) com 6 sensores.
Permite selecionar qual modelo usar via parametro 'modelo'.
"""
import json
import os
import joblib
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))

# Carrega artefatos compartilhados
scaler = joblib.load(os.path.join(_BASE, "..", "model", "scaler_falha.joblib"))
with open(os.path.join(_BASE, "..", "model", "feature_cols.json")) as f:
    feature_cols = json.load(f)

# Carrega mapa de nomes amigaveis (sensor_04 -> temp_turbina_c)
with open(os.path.join(_BASE, "..", "model", "feature_map.json")) as f:
    feature_map_raw = json.load(f)
# Inverte: temp_turbina_c -> sensor_04
friendly_to_sensor = {v["nome_amigavel"]: k for k, v in feature_map_raw.items()}

# Carrega ambos os modelos
xgb_model = joblib.load(os.path.join(_BASE, "..", "model", "xgboost_falha.joblib"))
rf_model = joblib.load(os.path.join(_BASE, "..", "model", "random_forest_falha.joblib"))

# Mapa de nomes amigaveis para os sensores
SENSOR_NAMES = {
    'sensor_04': 'Temperatura da Turbina',
    'sensor_07': 'Pressao do Compressor',
    'sensor_11': 'Pressao Estatica Saida',
    'sensor_12': 'Vazao de Combustivel',
    'sensor_15': 'Razao de Bypass',
    'sensor_21': 'Temperatura do Mancal'
}


def predict(features: dict, modelo: str = "xgboost") -> dict:
    """
    Recebe dicionario com 6 valores de sensores + modelo desejado.
    Retorna dict com classe, probabilidade_falha, status e mensagem.
    """
    try:
        # Valida modelo
        modelo = modelo.lower().strip()
        if modelo not in ("xgboost", "random_forest", "rf"):
            return {"status": "erro", "mensagem": f"Modelo invalido: {modelo}. Use 'xgboost' ou 'random_forest'."}

        # Extrai valores na ordem correta usando mapeamento
        # feature_map_raw: sensor_04 -> {nome_amigavel: temp_turbina_c, ...}
        valores = []
        for col_original in feature_cols:
            # friendly name como o usuario envia no JSON
            nome_friendly = feature_map_raw[col_original]["nome_amigavel"]
            valores.append(features[nome_friendly])
        X = np.array([valores])
        X_s = scaler.transform(X)

        # Seleciona modelo
        model = rf_model if modelo in ("random_forest", "rf") else xgb_model
        nome_modelo = "Random Forest" if modelo in ("random_forest", "rf") else "XGBoost"

        proba = model.predict_proba(X_s)[0, 1]
        classe = int(model.predict(X_s)[0])

        return {
            "classe": classe,
            "probabilidade_falha": round(float(proba), 4),
            "modelo": nome_modelo,
            "status": "falha" if classe == 1 else "normal",
            "mensagem": (
                f"[{nome_modelo}] Motor com FALHA IMINENTE! Realizar manutencao o quanto antes."
                if classe == 1
                else f"[{nome_modelo}] Motor operando em condicoes normais."
            ),
        }
    except KeyError as e:
        return {"status": "erro", "mensagem": f"Campo ausente: {e}"}
    except Exception as e:
        return {"status": "erro", "mensagem": str(e)}
