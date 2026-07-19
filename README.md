# CMAPSS Turbofan — Classificador de Falhas em Motores

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-green)](https://xgboost.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.108-teal)](https://fastapi.tiangolo.com)
[![Licença](https://img.shields.io/badge/Licen%C3%A7a-MIT-lightgrey)](LICENSE)

**Projeto Final — Disciplina de Machine Learning**  
Especialização em Inteligência Artificial

Dataset: [NASA CMAPSS FD001](https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6) (Turbofan Engine Degradation Simulation)

---

## Visão Geral

Sistema de classificação de falhas iminentes em motores turbofan, combinando **XGBoost** e **Random Forest** para prever se um motor apresentará falha nos próximos 30 ciclos de operação. O modelo foi treinado com o dataset CMAPSS da NASA e disponibilizado via API REST com interface web intuitiva.

**Destaques:**
- 94,81% de acurácia (Random Forest) com apenas **6 sensores**
- API FastASY com 3 endpoints
- Frontend web responsivo com 6 campos intuitivos
- Relatório acadêmico completo em PDF

---

## Dataset

### CMAPSS FD001 — NASA Turbofan Engine Degradation

O dataset simula a degradação de 100 motores turbofan ao longo de ciclos de operação, monitorados por **21 sensores** (temperatura, pressão, rotação, vibração, etc.) sob **3 condições operacionais**.

| Arquivo | Registros | Descrição |
|---------|-----------|-----------|
| `train_FD001.txt` | 20.631 | Ciclos de treino (100 motores) |
| `test_FD001.txt` | 13.096 | Ciclos de teste (100 motores, séries truncadas) |
| `RUL_FD001.txt` | 100 | Vida útil restante real para cada motor de teste |

### Target

**Falha Iminente (binário):** `1` se RUL ≤ 30 ciclos, `0` caso contrário.

Proporção no treino: 84,97% normal / 15,03% falha (desbalanceamento natural).

### Sensores Selecionados

Dos 21 sensores originais, selecionamos os **6 com maior correlação** com o target:

| Sensor | Nome na Interface | Correlação | Faixa Típica |
|--------|-------------------|:-----------:|--------------|
| sensor_04 | Temperatura da Turbina (°C) | 0,648 | 1380 — 1445 |
| sensor_07 | Pressão do Compressor (psi) | 0,626 | 549 — 557 |
| sensor_11 | Pressão Estática Saída (psi) | 0,666 | 46,5 — 49 |
| sensor_12 | Vazão de Combustível | 0,640 | 518 — 524 |
| sensor_15 | Razão de Bypass | 0,619 | 8,30 — 8,60 |
| sensor_21 | Temperatura do Mancal (°C) | 0,606 | 22,8 — 23,7 |

---

## Metodologia

### Fluxo do Projeto

```
Carregamento → EDA → Engenharia de Features → Split → 
StandardScaler → Treino (XGBoost + RF) → Avaliação → 
Exportação → API FastAPI → Frontend Web
```

### Modelos

| Modelo | Hiperparâmetros |
|--------|----------------|
| **XGBoost** | 150 estimadores, max_depth=6, learning_rate=0.1, scale_pos_weight |
| **Random Forest** | 100 estimadores, max_depth=15 |

### Pré-processamento
- Seleção das 6 features por correlação absoluta com o target
- Train/test split 80/20 com estratificação
- Padronização (StandardScaler)

---

## Resultados

### Tabela Comparativa

| Modelo | Acurácia | Precisão | Recall | F1-Score | AUC-ROC |
|--------|:--------:|:--------:|:------:|:--------:|:-------:|
| Random Forest | **94,81%** | 0,8512 | 0,7935 | **0,8214** | 0,9799 |
| XGBoost | 93,63% | 0,7309 | **0,9113** | 0,8112 | **0,9818** |

### Observações
- **Random Forest** — maior acurácia e F1-Score, melhor equilíbrio geral
- **XGBoost** — maior Recall (91%), detecta mais falhas reais (crucial para manutenção preditiva)
- Ambos com **AUC-ROC > 0,98** — excelente capacidade de discriminação
- Redução de 24 para 6 sensores **manteve** a qualidade preditiva

---

## Arquitetura

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│   Frontend  │────▶│  FastAPI     │────▶│   Modelo     │
│  index.html │     │  /predict    │     │  XGBoost     │
│  (6 campos) │◀────│  JSON        │◀────│  + Scaler    │
└─────────────┘     └──────────────┘     └──────────────┘
```

---

## Como Executar

### 1. Ambiente Virtual

```bash
python -m venv venv_projeto
source venv_projeto/Scripts/activate  # Windows
pip install -r requirements.txt
```

### 2. API

```bash
cd api
python main.py
# Servidor em http://localhost:8001
```

Endpoints:
- `GET /` — informações da API
- `GET /health` — status
- `POST /predict` — predição (6 sensores)

### 3. Frontend

Abra o arquivo `frontend/index.html` diretamente no navegador.

Os campos já vêm preenchidos com valores médios do dataset. Ajuste os parâmetros e clique em **"Analisar Motor"**.

### 4. Notebook

```bash
jupyter notebook notebooks/cmapss-classificacao.ipynb
```

### Exemplo de Requisição (API)

```json
POST /predict
{
  "temp_turbina_c": 1408.93,
  "pressao_compressor_psi": 553.37,
  "pressao_estatica_psi": 47.54,
  "vazao_combustivel": 521.41,
  "razao_bypass": 8.44,
  "temp_mancal_c": 23.29
}
```

Resposta:
```json
{
  "classe": 0,
  "probabilidade_falha": 0.0028,
  "status": "normal",
  "mensagem": "Motor operando em condições normais."
}
```

---

## Estrutura do Projeto

```
projeto_final/
├── data/
│   ├── train_FD001.txt          (20.631 ciclos)
│   ├── test_FD001.txt           (13.096 ciclos)
│   └── RUL_FD001.txt            (100 valores RUL)
├── notebooks/
│   └── cmapss-classificacao.ipynb    ← Notebook principal
├── model/
│   ├── xgboost_falha.joblib     (modelo treinado)
│   ├── scaler_falha.joblib      (padronizador)
│   ├── feature_cols.json        (lista de features)
│   └── feature_map.json         (mapeamento sensor ↔ nome)
├── api/
│   ├── main.py                  (FastAPI)
│   └── predictor_simplificado.py (predição)
├── frontend/
│   ├── index.html               (interface web)
│   ├── styles.css               (estilos)
│   └── script.js                (lógica)
├── report/
│   ├── relatorio_cmapss.pdf     (relatório 12 páginas)
│   ├── 01_distribuicao_target.png
│   ├── 02_degradacao_motor.png
│   ├── 03_matriz_correlacao.png
│   ├── 04_sensores_correlacao.png
│   ├── 05_tendencia_sensor.png
│   ├── 06_ciclos_ate_falha.png
│   ├── 07_matrizes_confusao.png
│   ├── 08_curva_roc.png
│   ├── 09_feature_importance_xgboost.png
│   ├── interface_principal.png
│   ├── interface_sucesso.png
│   └── interface_falha.png
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Tecnologias

| Categoria | Tecnologia |
|-----------|------------|
| Linguagem | Python 3.11 |
| Modelagem | XGBoost, scikit-learn (Random Forest, StandardScaler) |
| Análise | pandas, numpy, matplotlib, seaborn |
| API | FastAPI, uvicorn, Pydantic |
| Frontend | HTML5, CSS3, JavaScript (vanilla) |
| Versionamento | Git, GitHub |
| Documentação | fpdf2 |

---

## Licença

MIT © 2026 — Projeto acadêmico para fins educacionais.

Dataset CMAPSS fornecido pelo **NASA Prognostics Center of Excellence (PCoE)**.
