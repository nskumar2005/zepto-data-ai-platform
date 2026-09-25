"""FastAPI wrapper for the Zepto support assistant."""
from fastapi import FastAPI
from pydantic import BaseModel, Field
from .rag import Answer, ask

app=FastAPI(title="Zepto Support Assistant", version="1.0.0")

class AskRequest(BaseModel):
    query: str = Field(min_length=1)

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/ask",response_model=Answer)
def ask_endpoint(request: AskRequest):
    return ask(request.query)
