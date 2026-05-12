import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  BarChart3,
  Brain,
  CheckCircle2,
  Loader2,
  MessageSquareText,
  RefreshCw,
  Search,
  Server
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
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [query, setQuery] = useState({
    question: "Customer says the refund was approved but the money has not reached their bank account. What should I do?"
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
    await loadDashboard();
  }

  async function loadDashboard() {
    const healthData = await api("/api/health");
    setHealth(healthData);

    if (!healthData.ready) {
      setTickets([]);
      setInsights(null);
      setClusters([]);
      return;
    }

    const [ticketData, insightData, clusterData] = await Promise.all([
      api("/api/tickets?limit=20"),
      api("/api/insights"),
      api("/api/clusters")
    ]);

    setTickets(ticketData);
    setInsights(insightData);
    setClusters(clusterData);
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
          question: query.question
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
    loadDashboard().catch(() => {
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
          <a href="#insights">Insights</a>
          <a href="#tickets">Tickets</a>
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
            <p className="eyebrow">Live ticket dashboard</p>
            <h2>Support Operations Dashboard</h2>
          </div>
          <button className="icon-button" onClick={refreshHealth} title="Refresh status">
            <RefreshCw size={18} />
          </button>
        </header>

        {error && <div className="alert error">{error}</div>}
        {health?.loading && <div className="alert success">Loading the bundled ticket dataset...</div>}
        {health?.error && <div className="alert error">{health.error}</div>}

        <section id="overview" className="stats-grid">
          {stats.map((item) => (
            <div className="stat-card" key={item.label}>
              <item.icon size={20} />
              <span>{item.label}</span>
              <strong>{item.value}</strong>
            </div>
          ))}
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

          <div className="panel" id="tickets">
            <div className="panel-header">
              <div>
                <h3>Recent Tickets</h3>
                <p>Recent records from the bundled CSV.</p>
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
              <p>Ask a support question; the assistant retrieves similar tickets and recommends a solution.</p>
            </div>
          </div>

          <form className="assistant-form" onSubmit={handleRecommend}>
            <label className="question-field">
              Question
              <textarea
                rows="5"
                value={query.question}
                onChange={(event) => setQuery({ ...query, question: event.target.value })}
              />
            </label>

            <button type="submit" disabled={loading || !isReady}>
              {loading ? <Loader2 className="spin" size={17} /> : <Search size={17} />}
              Ask Assistant
            </button>
          </form>

          {recommendation && (
            <div className="recommendation">
              <h4>Recommended Solution</h4>
              <p>{recommendation.suggested_response}</p>
              <h4>Most Similar Tickets</h4>
              <Table
                rows={recommendation.matched_resolutions || recommendation.similar_tickets || []}
                columns={[
                  ["Ticket ID", "ID"],
                  ["Ticket Type", "Type"],
                  ["Ticket Priority", "Priority"],
                  ["Resolution", "Past Resolution"],
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
