"""
predictor.py
Carrega o modelo XGBoost treinado, o scaler e a lista de features,
e fornece a função predict() para classificação de falha em motores.
"""

import json
import os
import joblib
import numpy as np

# ── Caminhos dos artefatos ──────────────────────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH   = os.path.join(_BASE, "..", "model", "xgboost_falha.joblib")
_SCALER_PATH  = os.path.join(_BASE, "..", "model", "scaler_falha.joblib")
_FEATURES_PATH = os.path.join(_BASE, "..", "model", "feature_cols.json")

# ── Carga dos artefatos (uma vez na importação) ─────────────────────────
try:
    model = joblib.load(_MODEL_PATH)
except Exception as e:
    model = None
    _model_error = f"Erro ao carregar modelo: {e}"

try:
    scaler = joblib.load(_SCALER_PATH)
except Exception as e:
    scaler = None
    _scaler_error = f"Erro ao carregar scaler: {e}"

try:
    with open(_FEATURES_PATH, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)
except Exception as e:
    feature_cols = None
    _features_error = f"Erro ao carregar feature_cols.json: {e}"


def predict(features: dict) -> dict:
    """
    Executa a predição a partir de um dicionário de features.

    Parameters
    ----------
    features : dict
        Dicionário com os valores das features. As chaves devem
        corresponder aos nomes em ``feature_cols``.

    Returns
    -------
    dict
        Resultado da predição com as chaves:
        - classe              (int)
        - probabilidade_falha (float, 4 casas decimais)
        - status              (str)
        - mensagem            (str, condicional)
    """
    # ── Validação de carga ──────────────────────────────────────────────
    erros = []
    if model is None:
        erros.append("Modelo não carregado.")
    if scaler is None:
        erros.append("Scaler não carregado.")
    if feature_cols is None:
        erros.append("Lista de features não carregada.")

    if erros:
        return {
            "classe": -1,
            "probabilidade_falha": 0.0,
            "status": "erro",
            "mensagem": " | ".join(erros),
        }

    # ── Extrair valores na ordem correta ────────────────────────────────
    try:
        valores = [features[col] for col in feature_cols]
    except KeyError as e:
        return {
            "classe": -1,
            "probabilidade_falha": 0.0,
            "status": "erro",
            "mensagem": f"Feature ausente no dicionário de entrada: {e}",
        }

    X = np.array(valores).reshape(1, -1)

    # ── Scaler ──────────────────────────────────────────────────────────
    try:
        X_scaled = scaler.transform(X)
    except Exception as e:
        return {
            "classe": -1,
            "probabilidade_falha": 0.0,
            "status": "erro",
            "mensagem": f"Erro na transformação pelo scaler: {e}",
        }

    # ── Predição ────────────────────────────────────────────────────────
    try:
        proba = model.predict_proba(X_scaled)[0]
        classe = int(model.predict(X_scaled)[0])
    except Exception as e:
        return {
            "classe": -1,
            "probabilidade_falha": 0.0,
            "status": "erro",
            "mensagem": f"Erro na predição: {e}",
        }

    probabilidade_falha = round(float(proba[1]), 4)

    # ── Status e mensagem condicional ───────────────────────────────────
    if classe == 1:
        status = "falha"
        mensagem = "Motor classificado como FALHA iminente."
    else:
        status = "normal"
        mensagem = "Motor operando em condições normais."

    return {
        "classe": classe,
        "probabilidade_falha": probabilidade_falha,
        "status": status,
        "mensagem": mensagem,
    }
