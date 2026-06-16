# src/incident_copilot/llm.py
import os
from langchain_ollama import ChatOllama


def get_generation_llm() -> ChatOllama:
    model = os.getenv("GENERATION_LLM_MODEL", "gemma3:4b")
    return ChatOllama(model=model, temperature=0)


def get_fast_llm() -> ChatOllama:
    model = os.getenv("FAST_LLM_MODEL", "gemma3:1b")
    return ChatOllama(model=model, temperature=0)