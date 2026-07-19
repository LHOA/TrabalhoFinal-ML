"""
main.py
API FastAPI para predição de falha em motores (NASA Turbofan).
Usa o modelo simplificado de 6 sensores com nomes amigáveis.

Endpoints:
- GET  /        → mensagem de boas-vindas
- GET  /health  → status da API
- POST /predict → predição de falha (6 sensores)
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from predictor_simplificado import predict

# ── Instância da aplicação ──────────────────────────────────────────────
app = FastAPI(
    title="Falha em Motores - API Preditiva (Simplificada)",
    description=(
        "API para classificação de falha iminente em motores "
        "utilizando XGBoost treinado no dataset NASA Turbofan.\n"
        "Versão simplificada com 6 sensores e nomes amigáveis."
    ),
    version="2.0.0",
)

# ── CORS (permitir todas as origens) ────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Modelo Pydantic para os dados de entrada (6 sensores) ──────────────
class InputFeatures(BaseModel):
    temp_turbina_c: float = Field(
        ..., description="Temperatura da Turbina (1380-1440 °C)"
    )
    pressao_compressor_psi: float = Field(
        ..., description="Pressão do Compressor (549-556 psi)"
    )
    pressao_estatica_psi: float = Field(
        ..., description="Pressão Estática Saída (46.8-48.5 psi)"
    )
    vazao_combustivel: float = Field(
        ..., description="Vazão de Combustível (518-523)"
    )
    razao_bypass: float = Field(
        ..., description="Razão de Bypass (8.32-8.58)"
    )
    temp_mancal_c: float = Field(
        ..., description="Temperatura do Mancal (22.9-23.6 °C)"
    )


# ── Endpoints ───────────────────────────────────────────────────────────
@app.get("/")
def root():
    """Mensagem de boas-vindas da API."""
    return {
        "api": "Falha em Motores - API Preditiva (Simplificada)",
        "versao": "2.0.0",
        "status": "operacional",
        "endpoints": {
            "GET  /":        "esta mensagem",
            "GET  /health":  "verificação de saúde",
            "POST /predict": "predição de falha (6 sensores)",
        },
    }


@app.get("/health")
def health():
    """Verificação de saúde da API."""
    return {"status": "ok"}


@app.post("/predict")
def predict_endpoint(features: InputFeatures):
    """
    Recebe 6 features do motor com nomes amigáveis e retorna a classificação.
    """
    dados = features.model_dump()
    resultado = predict(dados)

    if resultado.get("status") == "erro":
        raise HTTPException(status_code=400, detail=resultado["mensagem"])

    return resultado


# ── Execução direta ────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001)
