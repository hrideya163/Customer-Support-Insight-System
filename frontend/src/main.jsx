import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  BarChart3,
  Brain,
  CheckCircle2,
  FileUp,
  Loader2,
  MessageSquareText,
  RefreshCw,
  Search,
  Server,
  Upload
} from "lucide-react";
import "./styles.css";

const API_BASE = "http://localhost:8000";

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(data.detail || "Request failed");
  }

  return data;
}

function App() {
  const [health, setHealth] = useState(null);
  const [tickets, setTickets] = useState([]);
  const [insights, setInsights] = useState(null);
  const [clusters, setClusters] = useState([]);
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [query, setQuery] = useState({
    subject: "refund not received",
    description: "customer says money never came back after refund was approved",
    top_k: 5
  });
  const [recommendation, setRecommendation] = useState(null);

  const isReady = Boolean(health?.ready);

  const stats = useMemo(() => {
    const common = insights?.most_common_issues?.[0];
    const frustration = insights?.high_frustration_categories?.[0];
    const slowest = insights?.slowest_resolutions?.[0];

    return [
      {
        label: "Rows Loaded",
        value: health?.rows_loaded ?? 0,
        icon: Server
      },
      {
        label: "Clusters",
        value: clusters.length,
        icon: BarChart3
      },
      {
        label: "Top Issue",
        value: common?.top_ticket_type || "Not ready",
        icon: Activity
      },
      {
        label: "Highest Risk",
        value: frustration?.["Ticket Type"] || slowest?.["Ticket Type"] || "Not ready",
        icon: Brain
      }
    ];
  }, [health, clusters, insights]);

  async function refreshHealth() {
    const data = await api("/api/health");
    setHealth(data);
  }

  async function loadDashboard() {
    const [healthData, ticketData, insightData, clusterData] = await Promise.all([
      api("/api/health"),
      api("/api/tickets?limit=20"),
      api("/api/insights"),
      api("/api/clusters")
    ]);

    setHealth(healthData);
    setTickets(ticketData);
    setInsights(insightData);
    setClusters(clusterData);
  }

  async function handleProcessDefault() {
    setLoading(true);
    setError("");
    setStatus("Processing default dataset...");

    try {
      const result = await api("/api/process", { method: "POST" });
      setStatus(result.message);
      await loadDashboard();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleUpload() {
    if (!file) {
      setError("Choose a CSV file first.");
      return;
    }

    setLoading(true);
    setError("");
    setStatus("Uploading and processing tickets...");

    try {
      const csvText = await file.text();
      const result = await api("/api/tickets/upload", {
        method: "POST",
        headers: { "Content-Type": "text/csv" },
        body: csvText
      });
      setStatus(result.message);
      await loadDashboard();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleRecommend(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setRecommendation(null);

    try {
      const result = await api("/api/tickets/recommend-response", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          subject: query.subject,
          description: query.description,
          top_k: Number(query.top_k) || 5
        })
      });
      setRecommendation(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refreshHealth().catch(() => {
      setHealth(null);
      setError("FastAPI backend is not reachable. Start it on http://localhost:8000.");
    });
  }, []);

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <MessageSquareText size={22} />
          </div>
          <div>
            <h1>Ticket Insights</h1>
            <p>Support analytics</p>
          </div>
        </div>

        <nav className="nav">
          <a href="#overview">Overview</a>
          <a href="#pipeline">Pipeline</a>
          <a href="#insights">Insights</a>
          <a href="#assistant">Assistant</a>
        </nav>

        <div className={`status-pill ${isReady ? "ready" : ""}`}>
          {isReady ? <CheckCircle2 size={16} /> : <Server size={16} />}
          {isReady ? "API ready" : "Waiting for data"}
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">AI processing pipeline</p>
            <h2>Customer Support Intelligence</h2>
          </div>
          <button className="icon-button" onClick={refreshHealth} title="Refresh status">
            <RefreshCw size={18} />
          </button>
        </header>

        {error && <div className="alert error">{error}</div>}
        {status && <div className="alert success">{status}</div>}

        <section id="overview" className="stats-grid">
          {stats.map((item) => (
            <div className="stat-card" key={item.label}>
              <item.icon size={20} />
              <span>{item.label}</span>
              <strong>{item.value}</strong>
            </div>
          ))}
        </section>

        <section id="pipeline" className="panel">
          <div className="panel-header">
            <div>
              <h3>Data Pipeline</h3>
              <p>Upload tickets or process the local 200k dataset.</p>
            </div>
          </div>

          <div className="actions-row">
            <label className="file-input">
              <FileUp size={18} />
              <span>{file ? file.name : "Choose CSV"}</span>
              <input type="file" accept=".csv" onChange={(event) => setFile(event.target.files?.[0])} />
            </label>

            <button onClick={handleUpload} disabled={loading}>
              <Upload size={17} />
              Upload
            </button>

            <button className="secondary" onClick={handleProcessDefault} disabled={loading}>
              {loading ? <Loader2 className="spin" size={17} /> : <Brain size={17} />}
              Process Default
            </button>
          </div>
        </section>

        <section id="insights" className="content-grid">
          <div className="panel wide">
            <div className="panel-header">
              <div>
                <h3>Business Insights</h3>
                <p>Generated from clusters, frustration signals, and resolution timing.</p>
              </div>
            </div>
            <div className="insight-text">
              {insights?.summary || "Process data to generate business insights."}
            </div>
          </div>

          <div className="panel">
            <div className="panel-header">
              <div>
                <h3>Recurring Issues</h3>
                <p>Top semantic clusters.</p>
              </div>
            </div>
            <Table
              rows={clusters.slice(0, 6)}
              columns={[
                ["issue_cluster", "Cluster"],
                ["ticket_count", "Tickets"],
                ["top_ticket_type", "Type"],
                ["top_terms", "Terms"]
              ]}
            />
          </div>

          <div className="panel">
            <div className="panel-header">
              <div>
                <h3>Recent Tickets</h3>
                <p>Sample records for the frontend.</p>
              </div>
            </div>
            <Table
              rows={tickets.slice(0, 8)}
              columns={[
                ["Ticket ID", "ID"],
                ["Ticket Type", "Type"],
                ["Ticket Priority", "Priority"],
                ["Customer Satisfaction Rating", "Rating"]
              ]}
            />
          </div>
        </section>

        <section id="assistant" className="panel assistant-panel">
          <div className="panel-header">
            <div>
              <h3>Response Assistant</h3>
              <p>Retrieve similar resolved tickets and draft a support response.</p>
            </div>
          </div>

          <form className="assistant-form" onSubmit={handleRecommend}>
            <label>
              Subject
              <input
                value={query.subject}
                onChange={(event) => setQuery({ ...query, subject: event.target.value })}
              />
            </label>

            <label>
              Description
              <textarea
                rows="4"
                value={query.description}
                onChange={(event) => setQuery({ ...query, description: event.target.value })}
              />
            </label>

            <label>
              Top K
              <input
                type="number"
                min="1"
                max="10"
                value={query.top_k}
                onChange={(event) => setQuery({ ...query, top_k: event.target.value })}
              />
            </label>

            <button type="submit" disabled={loading || !isReady}>
              {loading ? <Loader2 className="spin" size={17} /> : <Search size={17} />}
              Recommend Response
            </button>
          </form>

          {recommendation && (
            <div className="recommendation">
              <h4>Suggested Response</h4>
              <p>{recommendation.suggested_response}</p>
              <h4>Similar Tickets</h4>
              <Table
                rows={recommendation.similar_tickets || []}
                columns={[
                  ["Ticket ID", "ID"],
                  ["Ticket Type", "Type"],
                  ["Ticket Priority", "Priority"],
                  ["similarity_score", "Score"]
                ]}
              />
            </div>
          )}
        </section>
      </section>
    </main>
  );
}

function Table({ rows, columns }) {
  if (!rows?.length) {
    return <div className="empty-state">No data yet.</div>;
  }

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map(([, label]) => (
              <th key={label}>{label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={row["Ticket ID"] || row.issue_cluster || index}>
              {columns.map(([key]) => (
                <td key={key}>{formatValue(row[key])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function formatValue(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }

  if (typeof value === "number") {
    return Number.isInteger(value) ? value : value.toFixed(2);
  }

  return String(value);
}

createRoot(document.getElementById("root")).render(<App />);
