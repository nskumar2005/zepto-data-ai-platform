# Module 3 — Support Assistant

## Architecture

```text
8 policy docs
    ↓
ingest.py — one-document-per-chunk ingestion
    ↓
SentenceTransformer(all-MiniLM-L6-v2)
    ↓
ChromaDB collection: zepto_policy
    ↓
POST /ask
    ↓
LangGraph StateGraph
    ├── classify_intent
    │      ├── policy_question → retrieve_and_answer
    │      └── general_question → direct_answer
    ↓
Pydantic Answer(answer, sources, confidence)
```

**Ingestion:** `ingest.py` reads all eight `docs/doc_*.txt` files.

**Embedding:** `SentenceTransformer(all-MiniLM-L6-v2)` creates local embeddings; no embedding API key is required.

**Retrieval:** `retrieve_and_answer` embeds the incoming query and asks ChromaDB for the top three cosine-similar chunks.

**Generation:** with `MOCK_LLM` unset or `1`, `retrieve_and_answer` returns the required deterministic `Based on the retrieved context: ...` response and `direct_answer` returns the fixed policy-only response. With `MOCK_LLM=0`, the optional Groq path uses the structured prompt in `prompt.py` and retries validation up to two additional times.

## Run

```bash
python support_assistant/ingest.py
uvicorn support_assistant.main:app --reload --port 7860
```

Then POST JSON `{"query":"..."}` to `/ask`.

### Required mock examples

Policy-style example:

```json
{"query":"What is the standard delivery fee?"}
```

General example:

```json
{"query":"What is the capital of France?"}
```

### Actual local run responses

Policy-style example:

```json
{"answer":"Based on the retrieved context: Delivery Policy: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order vol","sources":["doc_01#0","doc_02#0","doc_05#0"],"confidence":1.0}
```

General example:

```json
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```

## Docker

From the repository root:

```bash
docker build -f support_assistant/Dockerfile -t zepto-support .
docker run --rm -p 7860:7860 zepto-support
```

## Optional real LLM

Set `MOCK_LLM=0` and `GROQ_API_KEY` only if you choose the optional extension. Never commit the API key. The graded baseline remains the deterministic mock path.
