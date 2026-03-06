# FinSense AI — Conversational Financial Clarity Engine

> **FinTech** · May – Aug 2024

## Overview

FinSense AI is a conversational GenAI engine that makes complex financial regulations accessible to everyday investors. Through **12 user interviews**, we discovered that retail investors struggle to understand SEBI (Securities and Exchange Board of India) regulatory documents, leading to poor financial decisions and compliance anxiety.

The system implements a **RAG pipeline over SEBI regulatory documents** with **confidence-scoring guardrails** and mapped **7 failure modes** to ensure safe, accurate responses. Task-completion rates improved from **34% → 91%** and the project won the **Best Product Concept** award.

## Problem Statement

| Metric | Before | After (Prototype) |
|--------|--------|-------------------|
| Task completion rate | 34% | 91% |
| Failure modes mapped | — | 7 |
| User interviews | — | 12 |
| Award | — | Best Product Concept |

## Architecture

```
User Query (Natural Language)
        │
        ▼
┌──────────────────────┐
│  Query Preprocessor  │  (Intent classification,
│                      │   entity extraction)
└──────────────────────┘
        │  processed_query
        ▼
┌──────────────────────┐
│  RAG Retriever       │  (Vector search over
│  (SEBI Documents)    │   embedded SEBI regulations)
└──────────────────────┘
        │  relevant_chunks[]
        ▼
┌──────────────────────┐
│  LLM Response        │  (Generate answer grounded in
│  Generator           │   retrieved SEBI context)
└──────────────────────┘
        │  draft_response
        ▼
┌──────────────────────┐
│  Confidence Scorer   │  (Score response reliability,
│  & Guardrails        │   flag low-confidence answers)
└──────────────────────┘
        │  scored_response
        ▼
┌──────────────────────┐
│  Failure Mode        │  (Check against 7 mapped
│  Detection           │   failure modes)
└──────────────────────┘
        │  safe_response
        ▼
┌──────────────────────┐
│  User-Friendly       │
│  Response            │
└──────────────────────┘
```

## Failure Modes

The system maps and guards against **7 identified failure modes**:

| # | Failure Mode | Mitigation |
|---|-------------|-------------|
| 1 | Hallucinated regulation | Strict RAG grounding + source citation |
| 2 | Outdated information | Document version tracking + staleness detection |
| 3 | Overly complex language | Readability scoring + plain-English rewrite |
| 4 | Misattributed source | Chunk-level provenance tracking |
| 5 | Incomplete answer | Coverage scoring against query entities |
| 6 | Confidential data leakage | PII detection + redaction layer |
| 7 | Misleading financial advice | Disclaimer injection + confidence thresholds |

## Setup

```bash
git clone https://github.com/tekigowtham2204/FinSense-AI.git
cd FinSense-AI
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env      # Add your OPENAI_API_KEY
uvicorn main:app --reload
```

API docs available at `http://localhost:8000/docs`

## Project Structure

```
finsense-ai/
├── main.py                       # FastAPI application entrypoint
├── config.py                     # Environment & model configuration
├── requirements.txt
├── .env.example
├── pipeline/
│   ├── query_preprocessor.py     # Intent classification & entity extraction
│   ├── rag_retriever.py          # Vector search over SEBI document embeddings
│   ├── response_generator.py     # LLM response generation with RAG context
│   ├── confidence_scorer.py      # Response confidence scoring & guardrails
│   ├── failure_detector.py       # 7-mode failure detection engine
│   └── embeddings.py             # Embedding utilities
├── models/
│   ├── query.py                  # Query & intent data models
│   ├── response.py               # Response & confidence data models
│   └── document.py               # SEBI document chunk models
├── data/
│   └── sebi_docs/                # SEBI regulatory documents (text/PDF)
├── api/
│   └── routes.py                 # API route definitions
└── tests/
    ├── test_retriever.py         # RAG retrieval tests
    ├── test_confidence.py        # Confidence scoring tests
    └── test_failure_modes.py     # Failure mode detection tests
```

## Tech Stack

| Layer | Technology |
|-------|------------|
| LLM | GPT-4 / Gemini Pro |
| Embeddings | OpenAI text-embedding-3-small |
| Vector store | ChromaDB |
| Backend | FastAPI (Python) |
| Orchestration | LangChain |
| Data models | Pydantic |
| Testing | pytest |

## Impact

- Task-completion rate improved from **34% → 91%**
- 7 failure modes mapped and mitigated with guardrails
- **Best Product Concept** award winner
- 12 user interviews drove product-market fit

## Author

**Gowtham Bhaskar Teki** — GenAI Product Manager
- LinkedIn: [linkedin.com/in/gowthambhaskar](https://linkedin.com/in/gowthambhaskar)
- GitHub: [github.com/tekigowtham2204](https://github.com/tekigowtham2204)
