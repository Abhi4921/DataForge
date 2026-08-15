# DataPilot

**DataPilot** is a GenAI-powered platform intended to help users move from project ideas to dataset discovery, evaluation, ranking, and recommendation.

> **Current Status:** Only the **Project Requirement Analyzer** module is implemented (v0.1.0). Dataset discovery, dataset ranking, dataset recommendation, dataset upload, frontend, database, RAG, and vector search are **NOT yet implemented**.

## Currently Implemented: Project Requirement Analyzer

The Project Requirement Analyzer takes a natural-language ML/AI project description (max 150 words) and converts it into structured requirements.

It combines:
1. **Deterministic keyword/concept dictionary matching** — a 147-concept knowledge base covering 22 domains
2. **Natural language understanding using Gemini** — structured LLM output via the Google GenAI SDK
3. **Dictionary + LLM reconciliation** — deduplication, provenance tracking
4. **Ambiguity detection** — does not hallucinate missing information
5. **Missing-information detection** — flags unclear objectives, missing targets
6. **Dataset requirement extraction** — describes what future datasets should contain

## Architecture

```
User Description
    ↓
FastAPI POST /api/v1/projects/analyze
    ↓
Word Count Validation (backend-enforced, 150-word limit)
    ↓
Dictionary Matching (deterministic, instant)
    ↓
Gemini LLM Analysis (semantic understanding, structured output)
    ↓
Reconciliation (merge, deduplicate, track provenance)
    ↓
Structured Response
```

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python 3.13+, FastAPI, Pydantic v2 |
| AI | Google GenAI SDK (`google-genai`) with Gemini 3.1 Flash Lite |
| Configuration | pydantic-settings, .env |
| Testing | pytest, httpx, FastAPI TestClient |
| Server | Uvicorn |

## Setup

### 1. Clone and enter the backend directory

```bash
git clone https://github.com/Sreeshan7/DataPilot.git
cd DataPilot/datapilot/backend
```

### 2. Create virtual environment

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and fill in your Gemini API key:

```bash
cp .env.example .env
```

Edit `.env`:
```
GEMINI_API_KEY=your_actual_api_key_here
GEMINI_MODEL=gemini-3.1-flash-lite
APP_ENV=development
LOG_LEVEL=INFO
```

### 5. Run the server

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 6. Access documentation

- **Swagger UI:** http://127.0.0.1:8000/docs
- **ReDoc:** http://127.0.0.1:8000/redoc
- **OpenAPI schema:** http://127.0.0.1:8000/openapi.json

## API Endpoints

### Health & Readiness

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check (no Gemini call) |
| GET | `/ready` | Readiness check (shows config state) |

### Project Analysis

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/projects/analyze` | Analyze a project description |

#### Request

```json
{
  "description": "I want to build a phishing detection system using machine learning on URL features."
}
```

#### Success Response

```json
{
  "success": true,
  "request_id": "uuid",
  "api_version": "v1",
  "data": {
    "input": { "word_count": 12 },
    "project_understanding": {
      "domain": { "value": "Cybersecurity", "confidence": 0.9 },
      "subdomain": { "value": "Web Security", "confidence": 0.85 },
      "problem_type": { "value": "Machine Learning", "confidence": 0.9 },
      "ml_tasks": [{ "task": "classification", "confidence": 0.87 }],
      "objective": "...",
      "target": { "value": "phishing/legitimate", "confidence": 0.7, "status": "inferred" }
    },
    "keywords": { "canonical": [...], "dictionary_matches": [...], "provenance": [...] },
    "dataset_requirements": { "feature_requirements": [...], ... },
    "analysis": { "overall_confidence": 0.85, "ambiguities": [...], "needs_clarification": false }
  },
  "meta": { "model": "gemini-3.1-flash-lite", "dictionary_version": "0.2.0", "processing_time_ms": 1234.5, "timestamp": "..." }
}
```

#### Error Response

```json
{
  "success": false,
  "error": {
    "code": "PROJECT_DESCRIPTION_TOO_LONG",
    "message": "Project description must not exceed 150 words.",
    "details": { "word_count": 151, "maximum_allowed": 150 }
  },
  "request_id": "uuid"
}
```

#### Error Codes

| Code | Description |
|------|-------------|
| `PROJECT_DESCRIPTION_REQUIRED` | Description field is missing |
| `PROJECT_DESCRIPTION_EMPTY` | Description is empty or whitespace only |
| `PROJECT_DESCRIPTION_TOO_LONG` | Exceeds 150-word limit |
| `LLM_AUTHENTICATION_ERROR` | Gemini API key is invalid |
| `LLM_RATE_LIMITED` | Gemini rate limit exceeded |
| `LLM_TIMEOUT` | Gemini request timed out |
| `LLM_UNAVAILABLE` | Gemini service unavailable |
| `LLM_INVALID_RESPONSE` | Gemini returned invalid structured data |
| `INTERNAL_ERROR` | Unexpected server error |

## Keyword Knowledge Base

The dictionary at `backend/data/keyword_dictionary.json` is a project-to-dataset knowledge base with **147 concepts** across **22 domains**:

- Artificial Intelligence (ML, Deep Learning, NLP, Computer Vision, Foundational ML)
- Cybersecurity (Network Security, Web Security, Fraud Detection)
- IoT (Internet of Things, Sensor Data, Edge Computing)
- Healthcare (Medical Imaging, Clinical Analysis, Genomics)
- Finance (Trading, Risk & Compliance)
- Education (Student Performance, Learning Analytics)
- Energy (Consumption, Forecasting)
- Agriculture, Environment, Manufacturing, Transportation
- Retail, Social Media, Sports, Real Estate, HR
- Telecom, Robotics, Operations Research
- Geospatial Analytics, Public Safety, Entertainment

Each concept includes: canonical term, synonyms, aliases, abbreviations, related concepts, ML tasks, dataset requirements, feature requirements, and target requirements.

## Testing

### Run unit tests

```bash
cd datapilot/backend
python -m pytest tests/unit/ -v
```

### Run with shorter output

```bash
python -m pytest tests/unit/ -v --tb=short
```

**Current test status:** 101 unit tests, all passing.

## Postman

Import the collection from:
```
postman/DataPilot-Project-Analyzer.postman_collection.json
```

Set the `base_url` variable to `http://localhost:8000`.

## Security

- API keys are stored in `.env` (never committed)
- `.env` is included in `.gitignore`
- API keys are never logged, printed, or returned in responses
- User descriptions are treated as untrusted data
- Prompt injection attempts are mitigated

## Currently NOT Implemented

- Dataset Discovery (Kaggle, HuggingFace, data.gov)
- Dataset Metadata Collection
- Dataset Quality Analysis
- Dataset Relevance Scoring
- Dataset Ranking
- Dataset Recommendation
- Dataset Upload & Analysis
- Frontend
- Database (PostgreSQL, etc.)
- RAG / Vector Search / Embeddings
- User Authentication

See `DEVELOPMENT_CONTEXT.md` for detailed technical documentation and the full roadmap.

## File Structure

```
DataPilot/
├── README.md
├── DEVELOPMENT_CONTEXT.md
├── .gitignore
├── datapilot/
│   ├── backend/
│   │   ├── app/
│   │   │   ├── __init__.py
│   │   │   ├── main.py
│   │   │   ├── api/v1/project_analysis.py
│   │   │   ├── core/config.py, logging_config.py
│   │   │   ├── schemas/project.py
│   │   │   ├── services/
│   │   │   │   ├── keyword_service.py
│   │   │   │   ├── llm_service.py
│   │   │   │   ├── reconciliation_service.py
│   │   │   │   └── project_analyzer.py
│   │   │   └── prompts/project_analysis.py
│   │   ├── data/keyword_dictionary.json
│   │   ├── tests/unit/
│   │   ├── requirements.txt
│   │   ├── .env.example
│   │   └── .env
│   └── postman/
│       └── DataPilot-Project-Analyzer.postman_collection.json
```
