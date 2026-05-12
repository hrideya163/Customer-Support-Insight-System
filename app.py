import os
import re
from io import StringIO

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


app = FastAPI(title="Ticket Insights API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


DATA_PATH = "customer_support_tickets_200k.csv"
MAX_ROWS = 200_000
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_API_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)


class TicketQuery(BaseModel):
    subject: str
    description: str
    top_k: int = 5


class PipelineState:
    def __init__(self):
        self.df = pd.DataFrame()
        self.embeddings = None
        self.embedding_model = None
        self.vectorizer = None
        self.tfidf_matrix = None
        self.faiss_index = None
        self.cluster_summary = pd.DataFrame()
        self.insights = {}
        self.ready = False


state = PipelineState()


def records(df):
    clean_df = df.replace({np.nan: None})
    return clean_df.to_dict(orient="records")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def yes_no_to_int(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0})
        .fillna(0)
        .astype(int)
    )


def prepare_tickets(df):
    column_map = {
        "ticket_id": "Ticket ID",
        "category": "Ticket Type",
        "issue_description": "Ticket Description",
        "resolution_notes": "Resolution",
        "priority": "Ticket Priority",
        "customer_satisfaction_score": "Customer Satisfaction Rating",
        "ticket_created_date": "Ticket Created Date",
        "ticket_resolved_date": "Ticket Resolved Date",
        "status": "Ticket Status",
        "product": "Product Purchased",
    }

    df = df.rename(columns=column_map).copy()

    if "Ticket Description" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must include issue_description or Ticket Description")

    if "Ticket Type" not in df.columns:
        df["Ticket Type"] = "General"

    if "Ticket Subject" not in df.columns:
        df["Ticket Subject"] = (
            df["Ticket Type"].fillna("Support issue").astype(str)
            + " - "
            + df["Ticket Description"].fillna("").astype(str).str.slice(0, 80)
        )

    defaults = {
        "Ticket ID": range(1, len(df) + 1),
        "Resolution": "",
        "Ticket Priority": "Medium",
        "Customer Satisfaction Rating": 3,
        "Ticket Created Date": pd.NaT,
        "Ticket Status": "",
        "Product Purchased": "",
        "resolution_time_hours": np.nan,
        "escalated": 0,
        "sla_breached": 0,
    }

    for column, value in defaults.items():
        if column not in df.columns:
            df[column] = value

    keep_columns = [
        "Ticket ID",
        "Ticket Subject",
        "Ticket Description",
        "Resolution",
        "Ticket Type",
        "Ticket Priority",
        "Customer Satisfaction Rating",
        "Ticket Created Date",
        "Ticket Status",
        "Product Purchased",
        "resolution_time_hours",
        "escalated",
        "sla_breached",
    ]
    df = df[keep_columns].copy()

    df["Customer Satisfaction Rating"] = pd.to_numeric(df["Customer Satisfaction Rating"], errors="coerce")
    df["resolution_time_hours"] = pd.to_numeric(df["resolution_time_hours"], errors="coerce")
    df["escalated"] = yes_no_to_int(df["escalated"])
    df["sla_breached"] = yes_no_to_int(df["sla_breached"])

    df["ticket_text"] = df["Ticket Subject"].fillna("") + " " + df["Ticket Description"].fillna("")
    df["clean_ticket_text"] = df["ticket_text"].apply(clean_text)
    return df


def build_embeddings(texts):
    if SentenceTransformer is not None:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        embeddings = model.encode(
            texts,
            batch_size=256,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return model, np.asarray(embeddings, dtype="float32"), None, None

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    return None, matrix.astype("float32"), vectorizer, matrix


def build_search_index(embeddings):
    if faiss is None or not isinstance(embeddings, np.ndarray):
        return None

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def search_similar(subject, description, top_k=5):
    if not state.ready:
        raise HTTPException(status_code=400, detail="No tickets loaded. Upload tickets or run /process first.")

    query = clean_text(f"{subject} {description}")

    if state.embedding_model is not None:
        query_embedding = state.embedding_model.encode([query], normalize_embeddings=True)
        query_embedding = np.asarray(query_embedding, dtype="float32")

        if state.faiss_index is not None:
            scores, indices = state.faiss_index.search(query_embedding, top_k)
            results = state.df.iloc[indices[0]].copy()
            results["similarity_score"] = scores[0]
            return results

        scores = cosine_similarity(query_embedding, state.embeddings).ravel()
    else:
        query_vector = state.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, state.tfidf_matrix).ravel()

    indices = np.argsort(scores)[::-1][:top_k]
    results = state.df.iloc[indices].copy()
    results["similarity_score"] = scores[indices]
    return results


def call_gemini_llm(prompt):
    if not GEMINI_API_KEY:
        return "GEMINI_API_KEY is not set, so no LLM response was generated."

    response = requests.post(
        GEMINI_API_URL,
        params={"key": GEMINI_API_KEY},
        headers={"Content-Type": "application/json"},
        json={
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 250,
            },
        },
        timeout=60,
    )
    response.raise_for_status()
    result = response.json()

    candidates = result.get("candidates", [])
    if not candidates:
        return "Gemini returned no response."

    parts = candidates[0].get("content", {}).get("parts", [])
    return " ".join(part.get("text", "") for part in parts).strip()


def create_cluster_summary(df, embeddings):
    n_clusters = min(12, max(2, len(df) // 1000))
    if len(df) < n_clusters:
        n_clusters = len(df)

    cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = cluster_model.fit_predict(embeddings if isinstance(embeddings, np.ndarray) else embeddings.toarray())
    df["issue_cluster"] = labels

    term_vectorizer = CountVectorizer(max_features=3000, stop_words="english", ngram_range=(1, 2))
    term_matrix = term_vectorizer.fit_transform(df["clean_ticket_text"])
    terms = np.array(term_vectorizer.get_feature_names_out())

    def top_terms(cluster_id):
        mask = df["issue_cluster"].values == cluster_id
        mean_terms = np.asarray(term_matrix[mask].mean(axis=0)).ravel()
        top_indices = mean_terms.argsort()[::-1][:8]
        return ", ".join(terms[top_indices])

    summary = (
        df.groupby("issue_cluster")
        .agg(
            ticket_count=("Ticket ID", "count"),
            top_ticket_type=("Ticket Type", lambda x: x.mode().iat[0]),
            avg_satisfaction=("Customer Satisfaction Rating", "mean"),
            avg_resolution_hours=("resolution_time_hours", "mean"),
            escalation_rate=("escalated", "mean"),
            sla_breach_rate=("sla_breached", "mean"),
        )
        .reset_index()
    )
    summary["top_terms"] = summary["issue_cluster"].apply(top_terms)
    return summary.sort_values("ticket_count", ascending=False)


def create_insights(df, cluster_summary):
    priority_score = df["Ticket Priority"].map({"Low": 1, "Medium": 2, "High": 3, "Critical": 4}).fillna(2)
    df["frustration_score"] = (
        (5 - df["Customer Satisfaction Rating"].fillna(3)) * 2
        + priority_score
        + df["escalated"] * 2
        + df["sla_breached"] * 2
    )

    high_frustration_categories = (
        df.groupby("Ticket Type")
        .agg(
            tickets=("Ticket ID", "count"),
            avg_frustration=("frustration_score", "mean"),
            avg_satisfaction=("Customer Satisfaction Rating", "mean"),
            escalation_rate=("escalated", "mean"),
        )
        .reset_index()
        .sort_values("avg_frustration", ascending=False)
    )

    slowest_resolutions = (
        df.groupby("Ticket Type")
        .agg(
            tickets=("Ticket ID", "count"),
            avg_resolution_hours=("resolution_time_hours", "mean"),
            sla_breach_rate=("sla_breached", "mean"),
        )
        .reset_index()
        .sort_values("avg_resolution_hours", ascending=False)
    )

    prompt = f"""
You are a support operations analyst.
Write 5 concise business insights from these tables.

Most common clusters:
{cluster_summary.head(5).to_string(index=False)}

High frustration categories:
{high_frustration_categories.head(5).to_string(index=False)}

Slowest resolution categories:
{slowest_resolutions.head(5).to_string(index=False)}
"""

    return {
        "summary": call_gemini_llm(prompt),
        "most_common_issues": records(cluster_summary.head(10)),
        "high_frustration_categories": records(high_frustration_categories.head(10)),
        "slowest_resolutions": records(slowest_resolutions.head(10)),
    }


def run_pipeline(df):
    prepared = prepare_tickets(df.head(MAX_ROWS))
    model, embeddings, vectorizer, tfidf_matrix = build_embeddings(prepared["clean_ticket_text"].tolist())
    index = build_search_index(embeddings)
    cluster_summary = create_cluster_summary(prepared, embeddings)
    insights = create_insights(prepared, cluster_summary)

    state.df = prepared
    state.embedding_model = model
    state.embeddings = embeddings
    state.vectorizer = vectorizer
    state.tfidf_matrix = tfidf_matrix
    state.faiss_index = index
    state.cluster_summary = cluster_summary
    state.insights = insights
    state.ready = True


@app.get("/")
def root():
    return {
        "message": "Ticket Insights API is running",
        "ready": state.ready,
        "rows_loaded": len(state.df),
    }


@app.post("/api/tickets/upload")
async def upload_tickets(request: Request):
    content = await request.body()
    if not content:
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    df = pd.read_csv(StringIO(content.decode("utf-8")))
    run_pipeline(df)

    return {
        "message": "Tickets uploaded and processed successfully.",
        "rows_loaded": len(state.df),
        "clusters": len(state.cluster_summary),
        "search_backend": "faiss" if state.faiss_index is not None else "cosine_similarity",
    }


@app.post("/api/process")
def process_default_dataset():
    if not os.path.exists(DATA_PATH):
        raise HTTPException(status_code=404, detail=f"{DATA_PATH} not found.")

    df = pd.read_csv(DATA_PATH, nrows=MAX_ROWS)
    run_pipeline(df)
    return {
        "message": "Default dataset processed successfully.",
        "rows_loaded": len(state.df),
        "clusters": len(state.cluster_summary),
        "search_backend": "faiss" if state.faiss_index is not None else "cosine_similarity",
    }


@app.get("/api/tickets")
def get_tickets(limit: int = 50):
    if not state.ready:
        raise HTTPException(status_code=400, detail="No tickets loaded.")

    columns = ["Ticket ID", "Ticket Subject", "Ticket Type", "Ticket Priority", "Customer Satisfaction Rating"]
    return records(state.df[columns].head(limit))


@app.post("/api/tickets/similar")
def get_similar_tickets(query: TicketQuery):
    results = search_similar(query.subject, query.description, query.top_k)
    return {"similar_tickets": records(results)}


@app.post("/api/tickets/recommend-response")
def recommend_response(query: TicketQuery):
    similar = search_similar(query.subject, query.description, query.top_k)
    resolutions = similar["Resolution"].dropna().astype(str)
    resolution_examples = "\n".join(f"- {text}" for text in resolutions.head(3))

    prompt = f"""
You are a customer support assistant.

New ticket subject: {query.subject}
New ticket description: {query.description}

Similar past resolutions:
{resolution_examples}

Write one concise suggested response for the support agent.
"""

    return {
        "suggested_response": call_gemini_llm(prompt),
        "similar_tickets": records(similar),
    }


@app.get("/api/insights")
def get_insights():
    if not state.ready:
        raise HTTPException(status_code=400, detail="No tickets loaded.")

    return state.insights


@app.get("/api/clusters")
def get_clusters():
    if not state.ready:
        raise HTTPException(status_code=400, detail="No tickets loaded.")

    return records(state.cluster_summary)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "ready": state.ready,
        "rows_loaded": len(state.df),
        "gemini_key_set": bool(GEMINI_API_KEY),
        "gemini_model": GEMINI_MODEL,
        "sentence_transformers_available": SentenceTransformer is not None,
        "faiss_available": faiss is not None,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
