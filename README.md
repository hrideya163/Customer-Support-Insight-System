# Ticket Insights Dashboard

A FastAPI and React dashboard for analyzing customer support tickets, surfacing recurring issue clusters, evaluating the ticket intelligence pipeline, and helping agents answer customer queries using similar historical cases.

## Features

- Automatically loads `customer_support_tickets_200k.csv` when the backend starts.
- Builds searchable ticket representations using Sentence Transformers when available, with TF-IDF fallback.
- Clusters recurring support issues with KMeans.
- Shows dashboard summaries for ticket volume, top issue types, frustration, and resolution risk.
- Displays model performance metrics from `pipeline_evaluation.csv`.
- Provides a Response Assistant where agents enter one support question and receive a recommended solution grounded in similar resolved tickets.
- Uses Gemini only for the Response Assistant. If the API key is missing or quota is exhausted, the app falls back to a retrieved-ticket based answer.

## Project Structure

```text
.
├── app.py                              # FastAPI backend
├── main.ipynb                          # Notebook pipeline, evaluation, and exports
├── customer_support_tickets_200k.csv   # Source ticket dataset
├── pipeline_evaluation.csv             # Model evaluation metrics shown in frontend
├── ticket_predictions.csv              # Notebook export
├── recurring_issue_clusters.csv        # Notebook export
└── frontend/
    ├── src/main.jsx                    # React dashboard
    ├── src/styles.css                  # Dashboard styling
    └── package.json                    # Vite scripts
```

## Backend Setup

Install Python dependencies as needed:

```powershell
pip install fastapi uvicorn pandas numpy scikit-learn requests
```

Optional, for semantic embeddings and faster similarity search:

```powershell
pip install sentence-transformers faiss-cpu
```

Run the backend from the project root:

```powershell
python app.py
```

The API runs at:

```text
http://127.0.0.1:8000
```

## Frontend Setup

From the `frontend` folder:

```powershell
npm install
npm run dev
```

Open the Vite URL, usually:

```text
http://localhost:5173
```

## Gemini Configuration

Gemini is optional and used only by the Response Assistant.

Create or update `.env` with:

```text
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

If no key is set, the assistant still returns a practical fallback based on the most similar historical resolved ticket.

## Main API Endpoints

- `GET /api/health` - backend readiness and model availability.
- `GET /api/tickets?limit=20` - recent ticket records.
- `GET /api/insights` - dashboard business insights.
- `GET /api/clusters` - recurring issue cluster summary.
- `GET /api/model-performance` - metrics from `pipeline_evaluation.csv`.
- `POST /api/tickets/recommend-response` - chatbot-style support assistant.

Example assistant request:

```json
{
  "question": "Customer says their refund was approved but money has not reached their bank. What should I do?"
}
```

## Notebook Workflow

Use `main.ipynb` to:

- Prepare and clean the ticket dataset.
- Build embeddings and similarity search.
- Generate issue clusters.
- Compute frustration signals.
- Export CSV files.
- Generate `pipeline_evaluation.csv` and matplotlib evaluation plots.

After rerunning the notebook, restart the backend so the dashboard picks up refreshed CSV outputs.
