/* ============================================
   CMAPSS Turbofan - Preditor de Falhas
   Script de Lógica e Integração com API
   ============================================ */

(function () {
    'use strict';

    /* --- DOM References --- */
    const form        = document.getElementById('predict-form');
    const btnAnalyze  = document.getElementById('btn-analyze');
    const btnReset    = document.getElementById('btn-reset');
    const btnText     = btnAnalyze.querySelector('.btn-text');
    const spinner     = btnAnalyze.querySelector('.spinner');
    const resultArea  = document.getElementById('result-area');
    const resultCard  = document.getElementById('result-card');
    const resultIcon  = document.getElementById('result-icon');
    const resultTitle = document.getElementById('result-title');
    const resultMsg   = document.getElementById('result-message');
    const resultProb  = document.getElementById('result-probability');
    const errorArea   = document.getElementById('error-area');
    const errorMsg    = document.getElementById('error-message');
    const btnNewDg    = document.getElementById('btn-new-diagnosis');

    const API_URL = 'http://localhost:8001/predict';

    /* --- Field definitions: id, label (for errors), min, max --- */
    const FIELDS = [
        { id: 'temp_turbina_c',        label: 'Temperatura da Turbina (°C)',   min: 1380, max: 1445 },
        { id: 'pressao_compressor_psi', label: 'Pressão do Compressor (psi)',   min: 549,  max: 557  },
        { id: 'pressao_estatica_psi',   label: 'Pressão Estática Saída (psi)',  min: 46.5, max: 49   },
        { id: 'vazao_combustivel',     label: 'Vazão de Combustível',          min: 518,  max: 524  },
        { id: 'razao_bypass',          label: 'Razão de Bypass',               min: 8.3,  max: 8.6  },
        { id: 'temp_mancal_c',         label: 'Temperatura do Mancal (°C)',    min: 22.8, max: 23.7 },
    ];

    /* --- Collect & Validate --- */
    function collectData() {
        const payload = {};
        const errors = [];

        FIELDS.forEach(function (field) {
            const input = document.getElementById(field.id);
            if (!input) {
                errors.push('Campo "' + field.label + '" não encontrado no formulário.');
                return;
            }

            const raw = input.value.trim();

            if (raw === '') {
                errors.push('O campo "' + field.label + '" está vazio.');
                input.classList.add('invalid');
                return;
            }

            const num = parseFloat(raw);

            if (isNaN(num)) {
                errors.push('O campo "' + field.label + '" contém valor inválido: "' + raw + '".');
                input.classList.add('invalid');
                return;
            }

            if (num < field.min || num > field.max) {
                errors.push('O campo "' + field.label + '" deve estar entre ' + field.min + ' e ' + field.max + '.');
                input.classList.add('invalid');
                return;
            }

            input.classList.remove('invalid');
            payload[field.id] = num;
        });

        return { payload: payload, errors: errors };
    }

    /* --- Clear validation styling --- */
    function clearValidation() {
        FIELDS.forEach(function (field) {
            const input = document.getElementById(field.id);
            if (input) input.classList.remove('invalid');
        });
    }

    /* --- Show / Hide Helpers --- */
    function showSuccess(probability, message) {
        resultArea.style.display = 'block';
        errorArea.style.display  = 'none';

        resultCard.className = 'result-card success';
        resultIcon.textContent = '✅';
        resultTitle.textContent = 'OPERACIONAL NORMAL';
        resultTitle.style.color = '';
        resultMsg.textContent   = message || 'O motor está operando dentro dos parâmetros esperados. Nenhuma falha iminente detectada.';
        resultProb.textContent  = (probability * 100).toFixed(2) + '%';
        resultProb.style.color  = '';

        resultArea.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    function showFailure(probability, message) {
        resultArea.style.display = 'block';
        errorArea.style.display  = 'none';

        resultCard.className = 'result-card failure';
        resultIcon.textContent = '⚠️';
        resultTitle.textContent = 'FALHA IMINENTE';
        resultTitle.style.color = '';
        resultMsg.textContent   = message || 'O motor apresenta sinais de falha iminente. Realizar manutenção o quanto antes.';
        resultProb.textContent  = (probability * 100).toFixed(2) + '%';
        resultProb.style.color  = '';

        resultArea.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    function showError(message) {
        errorArea.style.display  = 'block';
        resultArea.style.display = 'none';

        errorMsg.textContent = message;
        errorArea.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    function setLoading(isLoading) {
        btnAnalyze.disabled = isLoading;
        if (isLoading) {
            btnText.style.display = 'none';
            spinner.style.display = 'inline-block';
        } else {
            btnText.style.display = 'inline';
            spinner.style.display = 'none';
        }
    }

    function resetForm() {
        form.reset();
        resultArea.style.display = 'none';
        errorArea.style.display  = 'none';
        clearValidation();
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    /* --- API Call --- */
    async function callAPI(payload) {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            let detail = 'Erro do servidor (HTTP ' + response.status + ')';
            try {
                const errBody = await response.json();
                if (errBody.detail) detail = errBody.detail;
            } catch (_) {
                // ignore parse error
            }
            throw new Error(detail);
        }

        return await response.json();
    }

    /* --- Handle Prediction --- */
    async function handlePredict(event) {
        event.preventDefault();

        /* Hide previous results */
        resultArea.style.display = 'none';
        errorArea.style.display  = 'none';
        clearValidation();

        /* Collect and validate */
        const { payload, errors } = collectData();

        if (errors.length > 0) {
            showError('Por favor, corrija os erros abaixo:\\n- ' + errors.join('\\n- '));
            return;
        }

        /* Add selected model */
        const modeloSelect = document.getElementById('modelo-select');
        payload.modelo = modeloSelect ? modeloSelect.value : 'xgboost';

        /* Set loading */
        setLoading(true);

        try {
            const result = await callAPI(payload);

            const prediction = result.classe;
            const probability = result.probabilidade_falha;
            const isFalha = (prediction === 1 || prediction === '1' || prediction === true);

            if (isFalha) {
                showFailure(probability, result.mensagem);
            } else {
                showSuccess(probability, result.mensagem);
            }
        } catch (err) {
            let userMsg = 'Não foi possível conectar ao servidor de predição.';

            if (err.message && (
                err.message.includes('Failed to fetch') ||
                err.message.includes('NetworkError') ||
                err.message.includes('ERR_CONNECTION_REFUSED') ||
                err.message.includes('fetch')
            )) {
                userMsg = 'Servidor de predição offline. Verifique se a API está rodando em http://localhost:8000.';
            } else if (err.message) {
                userMsg = err.message;
            }

            showError('Erro na análise: ' + userMsg);
        } finally {
            setLoading(false);
        }
    }

    /* --- Event Listeners --- */
    form.addEventListener('submit', handlePredict);

    btnReset.addEventListener('click', resetForm);

    if (btnNewDg) {
        btnNewDg.addEventListener('click', resetForm);
    }

    /* Clear invalid class on input */
    FIELDS.forEach(function (field) {
        const input = document.getElementById(field.id);
        if (input) {
            input.addEventListener('input', function () {
                input.classList.remove('invalid');
            });
        }
    });

})();
