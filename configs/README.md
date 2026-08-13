# Configurações

Duas fontes de verdade, com responsabilidades separadas:

| Arquivo | Conteúdo | Exemplos |
|---------|----------|----------|
| `.env` | Infraestrutura e runtime | `API_KEY`, `INFERENCE_BACKEND`, `API_PORT` |
| `params.yaml` | Hiperparâmetros de ML | `max_features`, `test_size`, `n_estimators_baseline` |

Alterar `params.yaml` exige retreinar o modelo (`make train`).
