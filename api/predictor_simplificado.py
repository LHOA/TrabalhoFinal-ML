"""
predictor_simplificado.py
Carrega o modelo XGBoost simplificado (6 sensores) com nomes amigáveis.
Mapeia nomes amigáveis (ex: "temp_turbina_c") para nomes originais (ex: "sensor_04")
usando feature_map.json e extrai na ordem de feature_cols.json.
"""
import json
import os

import joblib
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))

# ── Carrega modelo, scaler, colunas e mapa ──────────────────────────────
model = joblib.load(os.path.join(_BASE, "..", "model", "xgboost_falha_simplificado.joblib"))
scaler = joblib.load(os.path.join(_BASE, "..", "model", "scaler_falha_simplificado.joblib"))

with open(os.path.join(_BASE, "..", "model", "feature_cols.json")) as f:
    feature_cols = json.load(f)

with open(os.path.join(_BASE, "..", "model", "feature_map.json")) as f:
    feature_map = json.load(f)

# feature_map: nome_original → nome_amigavel
# Ex: {"sensor_04": "temp_turbina_c", ...}


def predict(features: dict) -> dict:
    """
    Recebe um dicionário com chaves = nomes amigáveis (ex: "temp_turbina_c").
    Retorna dict com classe, probabilidade_falha, status e mensagem.
    """
    try:
        # Constrói vetor de features na ordem de feature_cols.json
        # usando o mapa para traduzir nome_original → nome_amigavel
        valores = []
        for col_original in feature_cols:
            nome_amigavel = feature_map[col_original]
            valores.append(features[nome_amigavel])

        X = np.array([valores])
        X_s = scaler.transform(X)

        proba = model.predict_proba(X_s)[0, 1]
        classe = int(model.predict(X_s)[0])

        return {
            "classe": classe,
            "probabilidade_falha": round(float(proba), 4),
            "status": "falha" if classe == 1 else "normal",
            "mensagem": (
                "Motor com FALHA IMINENTE! Realizar manutenção o quanto antes."
                if classe == 1
                else "Motor operando em condições normais."
            ),
        }
    except Exception as e:
        return {"status": "erro", "mensagem": str(e)}
