import { useState } from "react";
import axios from "axios";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
} from "recharts";

const API_BASE = "http://127.0.0.1:8000";

const COLORS = ["#6366f1", "#22c55e", "#f97316", "#ec4899", "#06b6d4"];

function App() {
  const [tickersInput, setTickersInput] = useState("AAPL,MSFT,SPY");
  const [horizon, setHorizon] = useState("6m");

  const [forecastLoading, setForecastLoading] = useState(false);
  const [optLoading, setOptLoading] = useState(false);

  const [forecastError, setForecastError] = useState("");
  const [optError, setOptError] = useState("");

  const [forecasts, setForecasts] = useState([]); // [{ticker, p10, p50, p90, horizon, spot, price_p50}]
  const [weights, setWeights] = useState(null); // { ticker: weight, ... }
  const [mu, setMu] = useState(null); // { ticker: expected_return, ... }

  const cleanTickers = () =>
    tickersInput
      .split(",")
      .map((t) => t.trim().toUpperCase())
      .filter(Boolean);

  // --------- API handlers ---------

  const handleForecast = async () => {
    setForecastError("");
    setOptError("");
    setForecastLoading(true);
    setWeights(null); // clear old optimization

    const tickersList = cleanTickers();
    if (tickersList.length === 0) {
      setForecastError("Please enter at least one ticker symbol.");
      setForecastLoading(false);
      return;
    }

    try {
      const params = new URLSearchParams({
        tickers: tickersList.join(","),
        horizon,
      }).toString();

      const res = await axios.get(`${API_BASE}/forecast?${params}`);
      const data = res.data?.forecasts || [];

      setForecasts(data);
    } catch (err) {
      console.error(err);
      setForecastError(
        err?.response?.data?.detail ||
          "Forecast request failed. Check backend logs or try different tickers."
      );
    } finally {
      setForecastLoading(false);
    }
  };

  const handleOptimize = async () => {
    setOptError("");
    setOptLoading(true);

    const tickersList = cleanTickers();
    if (tickersList.length === 0) {
      setOptError("Please enter at least one ticker symbol.");
      setOptLoading(false);
      return;
    }

    try {
      const res = await axios.post(`${API_BASE}/optimize`, {
        horizon,
        tickers: tickersList,
      });

      setWeights(res.data?.weights || null);
      setMu(res.data?.mu || null);
    } catch (err) {
      console.error(err);
      setOptError(
        err?.response?.data?.detail ||
          "Optimization failed. Check backend logs or try different tickers."
      );
    } finally {
      setOptLoading(false);
    }
  };

  // --------- derived data for charts ---------

  // Bar chart: current vs expected price
  const forecastChartData = forecasts.map((f) => ({
    ticker: f.ticker,
    spot: f.spot ?? null,
    expected: f.price_p50 ?? null,
  }));

  // Pie chart + table for weights
  const weightsChartData =
    weights &&
    Object.entries(weights).map(([ticker, w]) => ({
      name: ticker,
      value: w,
    }));

  const weightsTableData =
    weights &&
    Object.entries(weights).map(([ticker, w]) => ({
      ticker,
      weight: w,
      mu: mu && mu[ticker] != null ? mu[ticker] : null,
    }));

  // --------- UI ---------

  return (
    <div
      style={{
        minHeight: "100vh",
        background:
          "radial-gradient(circle at top, #020617, #020617 40%, #020617)",
        color: "#e5e7eb",
        fontFamily:
          "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        padding: "24px 24px 40px",
      }}
    >
      <div
        style={{
          maxWidth: "1280px",
          margin: "0 auto",
          display: "flex",
          flexDirection: "column",
          gap: "20px",
        }}
      >
        {/* HEADER */}
        <header
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-start",
            gap: "16px",
          }}
        >
          <div>
            <h1 style={{ fontSize: "30px", fontWeight: 700 }}>
              AI Portfolio Forecasting
            </h1>
            <p
              style={{
                color: "#9ca3af",
                marginTop: "4px",
                maxWidth: "640px",
                fontSize: "14px",
              }}
            >
              Ensemble of ElasticNet, LightGBM and LSTM with Modern Portfolio
              Theory and Yahoo Finance data. Forecast 6–12 months ahead and
              optimize portfolio weights.
            </p>
          </div>
          {/* Badge removed as requested */}
        </header>

        {/* INPUT CARD */}
        <section
          style={{
            background:
              "linear-gradient(135deg, rgba(15,23,42,0.95), rgba(15,23,42,0.9))",
            borderRadius: "18px",
            padding: "18px 22px",
            border: "1px solid rgba(148,163,184,0.25)",
            boxShadow: "0 18px 45px rgba(15,23,42,0.9)",
            display: "grid",
            gridTemplateColumns: "2fr 1.1fr",
            gap: "18px",
          }}
        >
          {/* Tickers + Horizon */}
          <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
            <label style={{ fontSize: "13px", color: "#9ca3af" }}>Tickers</label>
            <input
              value={tickersInput}
              onChange={(e) => setTickersInput(e.target.value)}
              placeholder="Example: AAPL,MSFT,SPY"
              style={{
                padding: "11px 13px",
                borderRadius: "10px",
                border: "1px solid #1f2937",
                background:
                  "linear-gradient(135deg, rgba(15,23,42,0.6), rgba(15,23,42,0.9))",
                color: "#e5e7eb",
                outline: "none",
                fontSize: "14px",
              }}
            />
            <span style={{ fontSize: "12px", color: "#6b7280" }}>
              Use comma-separated ticker symbols (US stocks / ETFs from Yahoo
              Finance).
            </span>

            <div
              style={{
                display: "flex",
                flexWrap: "wrap",
                gap: "12px",
                marginTop: "4px",
                alignItems: "center",
              }}
            >
              <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                <label style={{ fontSize: "13px", color: "#9ca3af" }}>
                  Forecast Horizon
                </label>
                <select
                  value={horizon}
                  onChange={(e) => setHorizon(e.target.value)}
                  style={{
                    padding: "8px 10px",
                    borderRadius: "10px",
                    border: "1px solid #1f2937",
                    background:
                      "linear-gradient(135deg, rgba(15,23,42,0.7), rgba(15,23,42,0.95))",
                    color: "#e5e7eb",
                    fontSize: "14px",
                  }}
                >
                  <option value="6m">6 months</option>
                  <option value="12m">12 months</option>
                </select>
              </div>
              <div
                style={{
                  fontSize: "12px",
                  color: "#6b7280",
                  maxWidth: "280px",
                  lineHeight: 1.4,
                }}
              >
                Median ensemble forecast over the selected horizon. Intervals
                (p10/p90) are experimental and may be N/A when data is limited.
              </div>
            </div>
          </div>

          {/* Actions */}
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              justifyContent: "space-between",
              gap: "12px",
            }}
          >
            <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
              <button
                onClick={handleForecast}
                disabled={forecastLoading}
                style={{
                  padding: "11px 16px",
                  borderRadius: "999px",
                  border: "none",
                  background: "linear-gradient(to right, #4f46e5, #22c55e)",
                  color: "white",
                  fontWeight: 600,
                  cursor: "pointer",
                  fontSize: "14px",
                  opacity: forecastLoading ? 0.7 : 1,
                  boxShadow: "0 10px 30px rgba(34,197,94,0.35)",
                }}
              >
                {forecastLoading ? "Running forecast..." : "Run Forecast"}
              </button>

              <button
                onClick={handleOptimize}
                disabled={optLoading}
                style={{
                  padding: "10px 16px",
                  borderRadius: "999px",
                  border: "1px solid rgba(148,163,184,0.45)",
                  background:
                    "linear-gradient(135deg, rgba(15,23,42,0.8), rgba(15,23,42,0.95))",
                  color: "#e5e7eb",
                  fontWeight: 500,
                  cursor: "pointer",
                  fontSize: "14px",
                  opacity: optLoading ? 0.7 : 1,
                }}
              >
                {optLoading ? "Optimizing..." : "Optimize Portfolio"}
              </button>
            </div>

            {(forecastError || optError) && (
              <div
                style={{
                  marginTop: "6px",
                  padding: "8px 10px",
                  borderRadius: "10px",
                  background:
                    "linear-gradient(135deg, rgba(127,29,29,0.2), rgba(127,29,29,0.4))",
                  border: "1px solid #b91c1c",
                  fontSize: "12px",
                  color: "#fee2e2",
                }}
              >
                <strong style={{ marginRight: "4px" }}>⚠</strong>
                {forecastError || optError}
              </div>
            )}

            <div
              style={{
                fontSize: "11px",
                color: "#6b7280",
                marginTop: "auto",
                borderTop: "1px dashed #1f2937",
                paddingTop: "8px",
              }}
            >
              Backend: FastAPI · Models: ElasticNet, LightGBM, LSTM · Optimization: max
              Sharpe using Ledoit–Wolf covariance.
            </div>
          </div>
        </section>

        {/* MAIN GRID: Forecasts + Charts */}
        <section
          style={{
            display: "grid",
            gridTemplateColumns: "1.4fr 1.1fr",
            gap: "18px",
          }}
        >
          {/* Forecast table */}
          <div
            style={{
              background: "rgba(15,23,42,0.95)",
              borderRadius: "18px",
              padding: "14px 18px",
              border: "1px solid rgba(30,64,175,0.45)",
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "baseline",
                marginBottom: "8px",
              }}
            >
              <h2 style={{ fontSize: "16px", fontWeight: 600 }}>Forecasts</h2>
              <span style={{ fontSize: "11px", color: "#9ca3af" }}>
                Prices plus log-returns over the selected horizon.
              </span>
            </div>

            {forecasts.length === 0 ? (
              <p
                style={{
                  fontSize: "13px",
                  color: "#6b7280",
                  marginTop: "4px",
                }}
              >
                Run a forecast to see model outputs for each ticker.
              </p>
            ) : (
              <table
                style={{
                  width: "100%",
                  borderCollapse: "collapse",
                  fontSize: "13px",
                  marginTop: "6px",
                }}
              >
                <thead>
                  <tr
                    style={{
                      borderBottom: "1px solid #1f2937",
                      color: "#9ca3af",
                    }}
                  >
                    <th style={{ textAlign: "left", padding: "6px 4px" }}>
                      Ticker
                    </th>
                    <th style={{ textAlign: "left", padding: "6px 4px" }}>
                      Horizon
                    </th>
                    <th style={{ textAlign: "right", padding: "6px 4px" }}>
                      Current price
                    </th>
                    <th style={{ textAlign: "right", padding: "6px 4px" }}>
                      Expected price (p50)
                    </th>
                    <th style={{ textAlign: "right", padding: "6px 4px" }}>
                      p50 (log)
                    </th>
                    <th
                      style={{
                        textAlign: "right",
                        padding: "6px 4px",
                        color: "#f97373",
                      }}
                    >
                      p10 (down)
                    </th>
                    <th
                      style={{
                        textAlign: "right",
                        padding: "6px 4px",
                        color: "#4ade80",
                      }}
                    >
                      p90 (up)
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {forecasts.map((f) => (
                    <tr
                      key={f.ticker}
                      style={{
                        borderBottom: "1px solid #020617",
                      }}
                    >
                      <td style={{ padding: "6px 4px", fontWeight: 600 }}>
                        {f.ticker}
                      </td>
                      <td style={{ padding: "6px 4px", color: "#9ca3af" }}>
                        {f.horizon}
                      </td>
                      <td style={{ padding: "6px 4px", textAlign: "right" }}>
                        {f.spot != null ? f.spot.toFixed(2) : "—"}
                      </td>
                      <td style={{ padding: "6px 4px", textAlign: "right" }}>
                        {f.price_p50 != null ? f.price_p50.toFixed(2) : "—"}
                      </td>
                      <td style={{ padding: "6px 4px", textAlign: "right" }}>
                        {f.p50 != null ? f.p50.toFixed(4) : "—"}
                      </td>
                      <td
                        style={{
                          padding: "6px 4px",
                          textAlign: "right",
                          color: "#f97373",
                        }}
                      >
                        {f.p10 != null ? f.p10.toFixed(4) : "N/A"}
                      </td>
                      <td
                        style={{
                          padding: "6px 4px",
                          textAlign: "right",
                          color: "#4ade80",
                        }}
                      >
                        {f.p90 != null ? f.p90.toFixed(4) : "N/A"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

          {/* Charts card */}
          <div
            style={{
              background: "rgba(15,23,42,0.95)",
              borderRadius: "18px",
              padding: "14px 18px",
              border: "1px solid rgba(30,64,175,0.45)",
              display: "flex",
              flexDirection: "column",
              gap: "14px",
            }}
          >
            <h2 style={{ fontSize: "16px", fontWeight: 600 }}>Visuals</h2>

            {/* Forecast bar chart */}
            <div style={{ height: "260px" }}>
              <p
                style={{
                  fontSize: "12px",
                  color: "#9ca3af",
                  marginBottom: "4px",
                }}
              >
                Expected price vs current price
              </p>
              {forecastChartData.length === 0 ? (
                <p style={{ fontSize: "12px", color: "#6b7280" }}>
                  Run forecast to populate this chart.
                </p>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={forecastChartData} barGap={4}>
                    <XAxis
                      dataKey="ticker"
                      tick={{ fill: "#9ca3af", fontSize: 11 }}
                    />
                    <YAxis
                      tick={{ fill: "#9ca3af", fontSize: 11 }}
                      tickFormatter={(v) => v.toFixed(0)}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "#020617",
                        border: "1px solid #374151",
                        borderRadius: "10px",
                        fontSize: "12px",
                      }}
                      formatter={(value) => [
                        value?.toFixed ? value.toFixed(2) : value,
                        
                      ]}
                    />
                    <Legend
                      wrapperStyle={{
                        fontSize: "11px",
                        color: "#e5e7eb",
                      }}
                    />
                    <Bar dataKey="spot" name="Current price" fill="#38bdf8" />
                    <Bar
                      dataKey="expected"
                      name="Expected price (p50)"
                      fill="#22c55e"
                    />
                  </BarChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>
        </section>

        {/* PORTFOLIO SECTION (NEW, FULL WIDTH) */}
        <section
          style={{
            marginTop: "20px",
            background: "rgba(255,255,255,0.03)",
            borderRadius: "18px",
            padding: "20px",
            border: "1px solid rgba(255,255,255,0.1)",
            backdropFilter: "blur(6px)",
          }}
        >
          <h2 style={{ fontSize: "16px", fontWeight: 600, marginBottom: "10px" }}>
            Portfolio Weights & Breakdown
          </h2>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: "20px",
              alignItems: "center",
            }}
          >
            {/* Pie chart */}
            <div style={{ height: "260px" }}>
              {!weightsChartData ? (
                <p style={{ fontSize: "13px", color: "#6b7280" }}>
                  Click &quot;Optimize Portfolio&quot; to see the optimal allocation.
                </p>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={weightsChartData}
                      dataKey="value"
                      nameKey="name"
                      innerRadius={45}
                      outerRadius={90}
                      paddingAngle={4}
                      label={({ name, value }) =>
                        `${name}: ${(value * 100).toFixed(1)}%`
                      }
                    >
                      {weightsChartData.map((entry, index) => (
                        <Cell
                          key={entry.name}
                          fill={COLORS[index % COLORS.length]}
                        />
                      ))}
                    </Pie>
                  </PieChart>
                </ResponsiveContainer>
              )}
            </div>

            {/* Weights table */}
            <div>
              {!weightsTableData ? (
                <p style={{ fontSize: "13px", color: "#6b7280" }}>
                  Run optimization to populate the weights table.
                </p>
              ) : (
                <table
                  style={{
                    width: "100%",
                    borderCollapse: "collapse",
                    fontSize: "13px",
                    background: "rgba(15,23,42,0.9)",
                    borderRadius: "12px",
                    overflow: "hidden",
                  }}
                >
                  <thead style={{ background: "rgba(15,23,42,0.95)" }}>
                    <tr style={{ color: "#d1d5db" }}>
                      <th style={{ textAlign: "left", padding: "8px" }}>Ticker</th>
                      <th style={{ textAlign: "right", padding: "8px" }}>Weight</th>
                      <th style={{ textAlign: "right", padding: "8px" }}>μ (log)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {weightsTableData.map((row) => (
                      <tr key={row.ticker}>
                        <td style={{ padding: "6px 8px", fontWeight: 600 }}>
                          {row.ticker}
                        </td>
                        <td style={{ padding: "6px 8px", textAlign: "right" }}>
                          {(row.weight * 100).toFixed(1)}%
                        </td>
                        <td
                          style={{
                            padding: "6px 8px",
                            textAlign: "right",
                            color: "#22c55e",
                          }}
                        >
                          {row.mu != null ? row.mu.toFixed(4) : "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

export default App;
