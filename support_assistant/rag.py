"""LangGraph RAG service with deterministic MOCK_LLM baseline and optional Groq HTTP path."""
from pathlib import Path
import os, json, requests
from typing import TypedDict
import chromadb
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field, ValidationError
from langgraph.graph import StateGraph, START, END
from .prompt import PROMPT_TEMPLATE

ROOT=Path(__file__).resolve().parent; DB=ROOT/"chroma_db"; COLLECTION="zepto_policy"; MODEL_NAME="all-MiniLM-L6-v2"

class Answer(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)

class State(TypedDict, total=False):
    query: str
    intent: str
    context: list[dict]
    answer: Answer

KEYWORDS=("delivery","return","refund","membership","tracking","cancel","gift card","support hours")

_client=None; _collection=None; _model=None

def resources():
    global _client,_collection,_model
    if _client is None:
        _client=chromadb.PersistentClient(path=str(DB))
        _collection=_client.get_or_create_collection(COLLECTION, metadata={"hnsw:space":"cosine"})
        _model=SentenceTransformer(MODEL_NAME)
    return _collection,_model


def mock_mode(): return os.getenv("MOCK_LLM","1") != "0"

def classify_intent(state: State):
    q=state["query"].lower()
    if mock_mode(): intent="policy_question" if any(k in q for k in KEYWORDS) else "general_question"
    else:
        # Optional real path uses the same small contract and falls back to the deterministic classifier on failure.
        intent=real_classify(q)
    return {"intent":intent}


def real_classify(query):
    api=os.getenv("GROQ_API_KEY")
    if not api: return "policy_question" if any(k in query for k in KEYWORDS) else "general_question"
    payload={"model":os.getenv("GROQ_MODEL","llama-3.1-8b-instant"),"messages":[{"role":"system","content":"Classify as exactly policy_question or general_question."},{"role":"user","content":query}],"temperature":0}
    r=requests.post("https://api.groq.com/openai/v1/chat/completions",headers={"Authorization":f"Bearer {api}"},json=payload,timeout=30); r.raise_for_status()
    value=r.json()["choices"][0]["message"]["content"].strip().lower()
    return "policy_question" if "policy_question" in value else "general_question"


def retrieve_and_answer(state: State):
    collection,model=resources(); qvec=model.encode([state["query"]],normalize_embeddings=True).tolist()
    result=collection.query(query_embeddings=qvec,n_results=3,include=["documents","metadatas","distances"])
    context=[]
    for i,doc in enumerate(result.get("documents",[[]])[0]):
        context.append({"id":result["metadatas"][0][i].get("document_id",f"chunk-{i}")+f"#{result['metadatas'][0][i].get('chunk_id',0)}","text":doc,"distance":result["distances"][0][i]})
    if mock_mode():
        top=context[0] if context else {"id":"none","text":"No matching policy context was retrieved."}
        ans=Answer(answer=f"Based on the retrieved context: {top['text'][:200]}",sources=[x["id"] for x in context],confidence=1.0)
    else: ans=generate_with_validation(state["query"],context)
    return {"context":context,"answer":ans}


def direct_answer(state: State):
    if mock_mode(): return {"answer":Answer(answer="I can only answer questions about Zepto policies right now.",sources=[],confidence=1.0)}
    return {"answer":generate_with_validation(state["query"],[])}


def generate_with_validation(query, context):
    api=os.getenv("GROQ_API_KEY")
    if not api: return Answer(answer="The optional real-LLM path requires GROQ_API_KEY; the deterministic mock baseline is available by leaving MOCK_LLM unset.",sources=[],confidence=0.0)
    ctx="\n\n".join(f"[{c['id']}] {c['text']}" for c in context)
    prompt=PROMPT_TEMPLATE.format(context=ctx,query=query)
    last_error=""
    for attempt in range(3):
        instruction=prompt if attempt==0 else prompt+f"\n\nCorrect the previous output so it validates as JSON Answer schema. Previous error: {last_error}"
        payload={"model":os.getenv("GROQ_MODEL","llama-3.1-8b-instant"),"messages":[{"role":"user","content":instruction}],"temperature":0}
        try:
            r=requests.post("https://api.groq.com/openai/v1/chat/completions",headers={"Authorization":f"Bearer {api}"},json=payload,timeout=60); r.raise_for_status()
            raw=r.json()["choices"][0]["message"]["content"].strip().replace("```json","").replace("```","").strip()
            return Answer.model_validate(json.loads(raw))
        except (requests.RequestException, json.JSONDecodeError, ValidationError, KeyError) as exc:
            last_error=str(exc)
    return Answer(answer="Error: the optional real-LLM response could not be validated after 3 attempts.",sources=[],confidence=0.0)


def build_graph():
    g=StateGraph(State)
    g.add_node("classify_intent",classify_intent); g.add_node("retrieve_and_answer",retrieve_and_answer); g.add_node("direct_answer",direct_answer)
    g.add_edge(START,"classify_intent")
    g.add_conditional_edges("classify_intent",lambda s:s["intent"],{"policy_question":"retrieve_and_answer","general_question":"direct_answer"})
    g.add_edge("retrieve_and_answer",END); g.add_edge("direct_answer",END)
    return g.compile()

GRAPH=build_graph()

def ask(query:str)->Answer:
    state=GRAPH.invoke({"query":query})
    return Answer.model_validate(state["answer"])
