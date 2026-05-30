# Источники (сиды для сборки)

Выверено веб-поиском 2026-05-30. Группировка по темам корпуса. Это стартовые
точки, не исчерпывающий список — Claude Code при сборке добавляет первоисточники
(arXiv, официальные доки) и проверяет факты минимум по двум источникам.

Помечено: [P] первоисточник/статья, [D] официальная документация,
[G] разбор/гайд, [V] волатильно (быстро устаревает, проверять дату).

---

## Раздел 1. Продукт (AI Product Engineer)

### 1.1 API провайдеров: structured outputs / tool use / streaming
- [P][D] OpenAI — Introducing Structured Outputs in the API
  https://openai.com/index/introducing-structured-outputs-in-the-api/
- [D] OpenAI — Function calling / Structured Outputs (docs)
- [D] Anthropic — Tool use (docs)
- [G][V] Structured outputs across OpenAI/Claude/Gemini (расхождения провайдеров)
  https://logic.inc/resources/structured-outputs-guide
- [G] Constrained decoding механика + overhead
  https://letsdatascience.com/blog/structured-outputs-making-llms-return-reliable-json
- [P] outlines / xgrammar / llguidance — движки constrained decoding (репозитории)

### 1.2 RAG: эмбеддинги / векторные БД / чанкинг / реранкеры / гибрид
- [P] Hybrid RRF + cross-encoder rerank, продакшен-пайплайн (репо-пример)
  https://github.com/tim-ponomarev/hybrid-rag
- [G] Hybrid search & re-ranking в проде, alpha-тюнинг, lost-in-the-middle
  https://towardsdatascience.com/hybrid-search-and-re-ranking-in-production-rag/
- [P] From BM25 to Corrective RAG (бенчмарк fusion CC vs RRF, глубина реранкера)
  https://arxiv.org/abs/2604.01733
- [G] Optimizing RAG with Hybrid Search & Reranking (формула BM25)
  https://superlinked.com/vectorhub/articles/optimizing-rag-with-hybrid-search-reranking
- [D] Qdrant docs; [D] pgvector (github.com/pgvector/pgvector)
- [P] MTEB leaderboard; BGE / E5 / GTE — семейства retrieval-эмбеддингов
- DSWoK §4.3 (RAG), §2.1 (двухбашенные), §3.2 (эмбеддинги)

### 1.3 Агентные системы на своём коде
- [P][D] Anthropic — Building Effective Agents (workflow vs agent, простота)
  https://www.anthropic.com/research/building-effective-agents
- [G] AI agent from scratch in Python (прямые API-вызовы, цикл)
  https://www.leoniemonigatti.com/blog/ai-agent-from-scratch-in-python.html
- [G] Production agent в ~70 строк на LiteLLM (мультипровайдер)
  https://newsletter.owainlewis.com/p/build-production-ai-agents-with-litellm
- [P] Yao et al. — ReAct (2022) arXiv:2210.03629

### 1.4 Evaluation: LLM-as-judge / ручные / регрессии
- [P] Zheng et al. — Judging LLM-as-a-Judge with MT-Bench, Chatbot Arena (2023)
  arXiv:2306.05685 (position/verbosity/self-preference bias)
- [D] RAGAS — Faithfulness / Context relevance / Answer relevance
- [G] Регрессионные наборы 10-20 кейсов, пороги вместо равенства
  https://www.braintrust.dev/articles/what-is-llm-monitoring
- DSWoK §4.3 (метрики retrieval и генерации)

### 1.5 Бэкенд: FastAPI / очереди / кэш / наблюдаемость
- [D] FastAPI; httpx (async); Pydantic
- [D] Celery / arq / RQ — очереди
- [D] Langfuse — LLM observability; [D] OpenTelemetry — спецификация трейсинга
- [G] OTel как vendor-neutral основа LLM-трейсинга
  https://www.braintrust.dev/articles/what-is-llm-monitoring

---

## Раздел 2. Модель (ML Engineer)

### 2.1 PyTorch (forward/backward, градиенты, loss curves)
- [D] PyTorch autograd mechanics (docs)
- [G] Karpathy — A Recipe for Training Neural Networks
  https://karpathy.github.io/2019/04/25/recipe/
- DSWoK §3.7 (vanishing/exploding gradients)

### 2.2 Файнтюн: SFT / LoRA-QLoRA / DPO / ORPO / KTO
- [P] Rafailov et al. — Direct Preference Optimization (2023) arXiv:2305.18290
- [P] Hong et al. — ORPO: Monolithic Preference Optimization (2024) arXiv:2403.07691
- [P] Dettmers et al. — QLoRA (2023) arXiv:2305.14314
- [P] Hu et al. — LoRA (2021) arXiv:2106.09685
- [G] DPO derivation from first principles (Bradley-Terry → loss)
  https://cameronrwolfe.substack.com/p/direct-preference-optimization
- [G] DPO/IPO/KTO/ORPO сравнение + код
  https://machinelearningplus.com/gen-ai/dpo-direct-preference-optimization/
- [G] Aman — Preference Optimization (DPO/KTO/PPO/GRPO primer)
  https://aman.ai/primers/ai/preference-optimization/
- [D] HuggingFace trl / peft; axolotl; unsloth
- DSWoK §2.7 (LoRA/QLoRA)

### 2.3 Подготовка данных: фильтрация / дедуп / разметка / синтетика
- [P] Wang et al. — Self-Instruct (2022) arXiv:2212.10560
- [P] Taori et al. — Stanford Alpaca (дистилляция)
- [G] MinHash + LSH для near-dedup на масштабе
- DSWoK §3.5 (fastText language id), §2.1 (эмбеддинги для семантич. дедупа)

### 2.4 Инференс: vLLM / llama.cpp / sglang, batching, KV-cache, квантизация
- [P] Kwon et al. — vLLM: Efficient Memory Management with PagedAttention
  (SOSP 2023) arXiv:2309.06180
- [G] vLLM + PagedAttention + continuous batching, продакшен
  https://www.runpod.io/articles/guides/vllm-pagedattention-continuous-batching
- [G][V] Квантизация на практике: формат-ядро-нагрузка (Marlin и т.д.)
  https://theaiengineer.substack.com/p/quantization-in-practice-gptq-vs
- [G][V] Бенчмарки квантизации в vLLM (Qwen2.5-32B, tok/s, Pass@1)
  https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks
- [G] GPTQ vs AWQ vs GGUF vs bnb — алгоритмы и выбор по стеку
  https://newsletter.maartengrootendorst.com/p/which-quantization-method-is-right
- [P] Frantar et al. — GPTQ; Lin et al. — AWQ; [D] llama.cpp (GGUF); [D] SGLang
- DSWoK §1.1.4-1.1.5 (MQA/GQA → размер KV-cache)

### 2.5 GPU-минимум: VRAM / FlashAttention / H100 vs 4090
- [P] Dao et al. — FlashAttention (2022) arXiv:2205.14135; FlashAttention-2 (2023)
- [D] NVIDIA H100 / Ada (RTX 4090) datasheets (bandwidth, FP8, NVLink)
- [G] Формулы VRAM под инференс/обучение/KV-cache
- DSWoK §1.1 (attention), §2.7 (LoRA-память)

---

## Раздел 3. Инфраструктура

### 3.1 Docker / Kubernetes / облако-bare-metal
- [D] NVIDIA Container Toolkit; Docker docs
- [D] Kubernetes docs; NVIDIA k8s device plugin (nvidia.com/gpu)
- [D] Документация выбранного облака (Yandex Cloud / VK Cloud GPU)

### 3.2 CI/CD / версионирование / эксперимент-трекинг
- [G][V] MLflow vs W&B vs Neptune — три философии (2025)
  https://uplatz.com/blog/the-2025-mlops-landscape-a-comparative-analysis-of-mlflow-weights-biases-and-neptune/
- [G] DVC vs MLflow vs W&B — стратегии версионирования
  https://mljourney.com/model-versioning-strategies-dvc-vs-mlflow-vs-weights-biases/
- [D] DVC (content-addressable, lakeFS-stewardship c 2025); [D] MLflow Model Registry
- [G] MLflow 3.0 — версионирование для GenAI/агентов (модель↔код↔промпт↔eval)
  https://introl.com/blog/model-versioning-infrastructure-mlops-artifact-management-guide-2025

### 3.3 Мониторинг в проде: дрейф данных/качества / стоимость / latency
- [G] Что такое LLM-мониторинг (quality/cost/latency/drift)
  https://www.braintrust.dev/articles/what-is-llm-monitoring
- [G] LLM drift как observability-, а не eval-проблема
  https://insightfinder.com/blog/hidden-cost-llm-drift-detection/
- [D] Evidently AI — data/prediction drift (PSI, KS); open-source
  https://www.evidentlyai.com/ml-in-production/model-monitoring
- [D] OpenTelemetry; Langfuse; MLflow AI monitoring
