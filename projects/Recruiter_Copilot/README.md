# Recruiter Copilot
Built a production-oriented recruiter copilot over 10,000 resumes with hybrid retrieval, reranking, LangGraph orchestration, LangSmith-based evaluation, and auditable shortlist generation with human review controls.

## Graph / Agent Flow — Baseline

```mermaid
%%{init: {"flowchart": {"htmlLabels": true}} }%%
flowchart LR
    START["<b>🚀 START</b>"] --> A["<b>Rewrite Query</b><br/>Reformulates recruiter query<br/>for better semantic retrieval"]
    A --> B["<b>Retrieve Relevant Resume Chunks</b><br/>Semantic search in PGVector over CV chunks<br/>Returns top-k chunks with metadata"]
    B --> C{"<b>Grade Retrieved Docs</b><br/>Are retrieved chunks relevant<br/>to the recruiter query?"}
    C -- ✅ Relevant --> D["<b>Generate Answer / Shortlist</b><br/>LLM creates grounded recruiter answer"]
    C -- ❌ Irrelevant --> E["<b>🔴 No Relevant Candidates Found</b><br/>END"]
    D --> F{"<b>Check Hallucination</b><br/>Is the answer supported by<br/>the retrieved chunks?"}
    F -- ✅ Grounded --> G["<b>✅ Answer Question</b><br/>END"]
    F -- ❌ Hallucinated --> H["<b>Regenerate Answer</b><br/>Retry with same evidence<br/>max 2 retries"]
    H --> F
```

## What to build  
The project should have these production-grade layers:
**Ingestion pipeline:** parse CVs, normalize structured fields, detect parsing errors, chunk documents intelligently, create embeddings, and attach metadata such as role, seniority, location, language, skills, and years of experience.
**Retrieval stack:** vector search, BM25 or keyword search, metadata filtering, then reranking to produce a top candidate set.
**Copilot layer:** LangGraph workflow for query understanding, retrieval, evidence aggregation, candidate comparison, shortlist drafting, and fallback behavior when evidence is weak.
**Evaluation layer:** golden datasets, retrieval evals, answer quality evals, failure slicing, trace review, and regression testing in LangSmith.
**Safety and compliance layer:** human-in-the-loop approval, explainable outputs, logging, and candidate-facing transparency assumptions documented clearly.