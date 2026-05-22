# Credit Risk RAG Service Architecture

This backend still runs as one FastAPI app by default, but it now has explicit
service boundaries for a split deployment.

## Service Boundaries

- API gateway: `services.api_gateway:app` on port `8000`.
- LLM inference service: `services.inference_service:app` on port `8011`.
- Retrieval service: `services.retrieval_service:app` on port `8012`.
- Tabular scoring service: `services.tabular_scoring_service:app` on port `8013`.

Local commands:

```powershell
cd BACKEND
uvicorn services.api_gateway:app --host 0.0.0.0 --port 8000
uvicorn services.inference_service:app --host 0.0.0.0 --port 8011
uvicorn services.retrieval_service:app --host 0.0.0.0 --port 8012
uvicorn services.tabular_scoring_service:app --host 0.0.0.0 --port 8013
```

## External LLM Model Server

Set `LLM_MODEL_SERVER_URL` to route generation away from local Transformers.
The client supports OpenAI-compatible vLLM by default and TGI with
`LLM_MODEL_SERVER_TYPE=tgi`.

```powershell
$env:LLM_MODEL_SERVER_URL="http://localhost:8011"
$env:LLM_MODEL_SERVER_TYPE="openai"
$env:LLM_MODEL_SERVER_MODEL="meta-llama/Llama-3.2-1B-Instruct"
```

For vLLM, point at the server base URL. For example, if vLLM exposes
`/v1/chat/completions`, use the base URL or the full chat completion URL.

## Job Queue

The default queue is in-process and dependency-free:

- `POST /jobs/ingest`
- `POST /jobs/eval/rag`
- `POST /jobs/batch-score`
- `GET /jobs`
- `GET /jobs/{job_id}`

The facade lives in `job_queue.py`. It is intentionally shaped so Celery, RQ, or
Temporal can replace the in-process executor without changing route contracts.

## Feature Store Contract

`feature_store.py` owns the canonical tabular feature contract:

- `GET /features/contract`
- `POST /features/transform`

Training and inference should both use this module so one-hot encoding, numeric
bounds, categorical normalization, and derived metrics stay consistent.

Set `TABULAR_SCORING_SERVICE_URL` on the gateway to delegate `/predict`,
scenario scoring, and queued batch scoring to the standalone scoring service:

```powershell
$env:TABULAR_SCORING_SERVICE_URL="http://localhost:8013"
```

## Event And Metrics Pipeline

`event_bus.py` publishes structured JSON events to stdout. If
`OTEL_EXPORTER_OTLP_ENDPOINT` is set, it also sends best-effort OTLP HTTP JSON
logs to an OpenTelemetry Collector:

```powershell
$env:OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318"
$env:SERVICE_NAME="credit-risk-rag-api"
```

Existing `/metrics/summary` now includes queue status and service topology.
