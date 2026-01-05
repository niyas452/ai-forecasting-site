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
  LineChart,
  Line,
  CartesianGrid,
  ReferenceDot,
} from "recharts";

const API_BASE = "http://127.0.0.1:8000";

const COLORS = ["#6366f1", "#22c55e", "#f97316", "#ec4899", "#06b6d4"];

// Helper to map YYYY-MM strings to linear indices
const ymToIndex = (ym) => {
  const [y, m] = ym.split("-").map(Number);
  return y * 12 + (m - 1);
};

const indexToYM = (idx) => {
  const y = Math.floor(idx / 12);
  const m = (idx % 12) + 1;
  return `${y}-${String(m).padStart(2, "0")}`;
};

// Generates the 5-point visual path: History -> Spot -> Forecast.
// Used for the sparkline-style charts.
const buildFivePointSeries = (forecastItem, horizon) => {
  const series = Array.isArray(forecastItem?.chart_series)
    ? forecastItem.chart_series
    : [];
  if (series.length === 0) return [];

  const actual = series.filter((p) => p.kind === "actual");
  const forecast = series.find((p) => p.kind === "forecast") || null;

  if (actual.length === 0) return [];

  const lastActual = actual[actual.length - 1]; // now/current month-end
  const nowYM = lastActual.date;
  const nowIdx = ymToIndex(nowYM);

  const step = horizon === "12m" ? 12 : 6;
  const offsets = [-3 * step, -2 * step, -1 * step, 0]; // 4 historical incl now
  const targetIdx = offsets.map((o) => nowIdx + o);

  // map actual date -> price
  const actualMap = new Map(actual.map((p) => [p.date, p.price]));

  // If exact month missing, pick nearest available actual month (closest by idx)
  const findNearestActualYM = (wantedIdx) => {
    let best = actual[0]?.date;
    let bestDist = Infinity;
    for (const p of actual) {
      const d = Math.abs(ymToIndex(p.date) - wantedIdx);
      if (d < bestDist) {
        bestDist = d;
        best = p.date;
      }
    }
    return best;
  };

  const points = targetIdx.map((idx) => {
    const ym = indexToYM(idx);
    const exact = actualMap.get(ym);
    if (exact != null) {
      return { date: ym, price: exact, kind: "actual" };
    }
    const nearestYM = findNearestActualYM(idx);
    return {
      date: nearestYM,
      price: actualMap.get(nearestYM),
      kind: "actual",
    };
  });

  // ensure unique dates in case nearest selection duplicates
  const unique = [];
  const seen = new Set();
  for (const p of points) {
    if (!seen.has(p.date)) {
      seen.add(p.date);
      unique.push(p);
    }
  }

  // add forecast point (final point)
  if (forecast && forecast.price != null && forecast.date) {
    unique.push({ date: forecast.date, price: forecast.price, kind: "forecast" });
  }

  return unique;
};

// Formatter for table values (handles null/zero robustness)
const fmtInterval = (v) => {
  if (v === null || v === undefined) return "N/A";
  const n = Number(v);
  if (!Number.isFinite(n)) return "N/A";
  if (Math.abs(n) < 1e-6) return "N/A"; // hide 0.0000 / -0.0000
  return n.toFixed(4);
};

function App() {
  const [tickersInput, setTickersInput] = useState("");
  const [horizon, setHorizon] = useState("6m");

  const [forecastLoading, setForecastLoading] = useState(false);
  const [optLoading, setOptLoading] = useState(false);

  const [forecastError, setForecastError] = useState("");
  const [optError, setOptError] = useState("");

  const [forecasts, setForecasts] = useState([]); // includes chart_series now
  const [weights, setWeights] = useState(null);
  const [mu, setMu] = useState(null);

  const cleanTickers = () =>
    tickersInput
      .split(",")
      .map((t) => t.trim().toUpperCase())
      .filter(Boolean);

  // API Interaction

  const handleForecast = async () => {
    setForecastError("");
    setOptError("");
    setForecastLoading(true);
    setWeights(null);

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

  // Data Transformation

  const forecastChartData = forecasts.map((f) => ({
    ticker: f.ticker,
    spot: f.spot ?? null,
    expected: f.price_p50 ?? null,
  }));

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

  // Map raw forecasts to charting structures
  const fivePointSeriesByTicker = forecasts.map((f) => ({
    ticker: f.ticker,
    series: buildFivePointSeries(f, horizon),
  }));

  // Custom renderer to selectively hide/show points on the line chart
  const CustomDot = (props) => {
    const { cx, cy, payload } = props;
    if (!payload) return null;

    // Skip forecast point (drawn separately)
    if (payload.kind === "forecast") return null;

    // blue dot for actual points
    return <circle cx={cx} cy={cy} r={3} fill="#38bdf8" />;
  };

  // Render Logic

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
        {/* Header */}
        <header
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-start",
            gap: "16px",
          }}
        >
          <div>
            <h1 style={{ fontSize: "42px", fontWeight: 700 }}>
              AI Portfolio Forecasting
            </h1>
            <p
              style={{
                color: "#cbd5e1",
                marginTop: "4px",
                maxWidth: "700px",
                fontSize: "18px",
              }}
            >
              AI-Powered Ensemble Forecasting & Portfolio Optimization (6–12m).
            </p>
          </div>
        </header>

        {/* Configuration Panel */}
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
            <label style={{ fontSize: "18px", color: "#e5e7eb", fontWeight: 600 }}>Tickers</label>
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
                color: "#ffffff",
                outline: "none",
                fontSize: "18px",
              }}
            />
            <span style={{ fontSize: "15px", color: "#cbd5e1" }}>
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
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: "4px",
                }}
              >
                <label style={{ fontSize: "18px", color: "#e5e7eb", fontWeight: 600 }}>
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
                    color: "#ffffff",
                    fontSize: "18px",
                  }}
                >
                  <option value="6m">6 months</option>
                  <option value="12m">12 months</option>
                </select>
              </div>
              <div
                style={{
                  fontSize: "15px",
                  color: "#cbd5e1",
                  maxWidth: "280px",
                  lineHeight: 1.5,
                }}
              >
                Median forecast. Intervals (p10/p90) are experimental.
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
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: "10px",
              }}
            >
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
                  fontSize: "16px",
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
                  fontSize: "16px",
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
                  fontSize: "14px",
                  color: "#fee2e2",
                }}
              >
                <strong style={{ marginRight: "4px" }}>⚠</strong>
                {forecastError || optError}
              </div>
            )}


          </div>
        </section>

        {/* Results Dashboard */}
        <section
          style={{
            display: "grid",
            gridTemplateColumns: "1.4fr 1.1fr",
            gap: "18px",
          }}
        >
          {/* Tabular Data */}
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
              <h2 style={{ fontSize: "18px", fontWeight: 600 }}>Forecasts</h2>
              <span style={{ fontSize: "16px", color: "#cbd5e1" }}>
                Prices plus log-returns over the selected horizon.
              </span>
            </div>

            {forecasts.length === 0 ? (
              <p style={{ fontSize: "15px", color: "#bdc6d4", marginTop: "4px" }}>
                Run a forecast to see model outputs for each ticker.
              </p>
            ) : (
              <table
                style={{
                  width: "100%",
                  borderCollapse: "collapse",
                  fontSize: "17px",
                  marginTop: "6px",
                }}
              >
                <thead>
                  <tr
                    style={{
                      borderBottom: "1px solid #1f2937",
                      color: "#e5e7eb",
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
                      style={{ borderBottom: "1px solid #020617" }}
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
                      {/* Formatting for zero values */}
                      <td
                        style={{
                          padding: "6px 4px",
                          textAlign: "right",
                          color: "#f97373",
                        }}
                      >
                        {fmtInterval(f.p10)}
                      </td>
                      <td
                        style={{
                          padding: "6px 4px",
                          textAlign: "right",
                          color: "#4ade80",
                        }}
                      >
                        {fmtInterval(f.p90)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

          {/* Summary Charts */}
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
            <h2 style={{ fontSize: "18px", fontWeight: 600 }}>Visuals</h2>

            <div style={{ height: "260px" }}>
              <p
                style={{
                  fontSize: "16px",
                  color: "#cbd5e1",
                  marginBottom: "4px",
                }}
              >
                Expected price vs current price
              </p>
              {forecastChartData.length === 0 ? (
                <p style={{ fontSize: "16px", color: "#9ca3af" }}>
                  Run forecast to populate this chart.
                </p>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={forecastChartData} barGap={4}>
                    <XAxis
                      dataKey="ticker"
                      tick={{ fill: "#e5e7eb", fontSize: 15 }}
                    />
                    <YAxis
                      tick={{ fill: "#e5e7eb", fontSize: 15 }}
                      tickFormatter={(v) => v.toFixed(0)}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "#020617",
                        border: "1px solid #374151",
                        borderRadius: "10px",
                        fontSize: "16px",
                      }}
                      formatter={(value) => [
                        value?.toFixed ? value.toFixed(2) : value,
                      ]}
                    />
                    <Legend
                      wrapperStyle={{ fontSize: "15px", color: "#e5e7eb" }}
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

        {/* Detailed Price Paths */}
        <section
          style={{
            marginTop: "4px",
            background: "rgba(255,255,255,0.03)",
            borderRadius: "18px",
            padding: "20px",
            border: "1px solid rgba(255,255,255,0.1)",
            backdropFilter: "blur(6px)",
          }}
        >
          <h2 style={{ fontSize: "18px", fontWeight: 600, marginBottom: "10px" }}>
            Price Path (5 points)
          </h2>
          <p style={{ fontSize: "16px", color: "#cbd5e1", marginBottom: "14px" }}>
            {horizon === "6m"
              ? "Points are spaced by 6 months (past 18m → now → forecast)."
              : "Points are spaced by 12 months (past 36m → now → forecast)."}{" "}
            Forecast point is highlighted in red.
          </p>

          {forecasts.length === 0 ? (
            <p style={{ fontSize: "13px", color: "#6b7280" }}>
              Run forecast to display line charts.
            </p>
          ) : (
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(360px, 1fr))",
                gap: "16px",
              }}
            >
              {fivePointSeriesByTicker.map(({ ticker, series }) => {
                const forecastPoint = series.find((p) => p.kind === "forecast");

                const lastActualIdx = (() => {
                  for (let i = series.length - 1; i >= 0; i--) {
                    if (series[i]?.kind === "actual") return i;
                  }
                  return -1;
                })();

                const chartData = series.map((p, i) => ({
                  date: p.date,
                  kind: p.kind,
                  actual: p.kind === "actual" ? p.price : null,
                  connector:
                    i === lastActualIdx || p.kind === "forecast" ? p.price : null,
                }));

                return (
                  <div
                    key={ticker}
                    style={{
                      background: "rgba(15,23,42,0.9)",
                      borderRadius: "14px",
                      padding: "12px 14px",
                      border: "1px solid rgba(30,64,175,0.35)",
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        marginBottom: "6px",
                      }}
                    >
                      <div style={{ fontWeight: 700 }}>{ticker}</div>
                      <div style={{ fontSize: "15px", color: "#9ca3af" }}>
                        {horizon}
                      </div>
                    </div>

                    <div style={{ height: "220px" }}>
                      {series.length === 0 ? (
                        <p style={{ fontSize: "15px", color: "#6b7280" }}>
                          No chart data available.
                        </p>
                      ) : (
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={chartData}>
                            <CartesianGrid stroke="#1f2937" strokeDasharray="3 3" />
                            <XAxis
                              dataKey="date"
                              tick={{ fill: "#e5e7eb", fontSize: 15 }}
                            />
                            <YAxis
                              tick={{ fill: "#e5e7eb", fontSize: 15 }}
                              tickFormatter={(v) => Number(v).toFixed(0)}
                              domain={["auto", "auto"]}
                            />
                            <Tooltip
                              contentStyle={{
                                background: "#020617",
                                border: "1px solid #374151",
                                borderRadius: "10px",
                                fontSize: "16px",
                              }}
                              formatter={(value, name, props) => {
                                const kind = props?.payload?.kind;
                                const label =
                                  kind === "forecast" ? "Forecast" : "Actual";
                                return [Number(value).toFixed(2), label];
                              }}
                            />

                            {/* Historical trajectory */}
                            <Line
                              type="monotone"
                              dataKey="actual"
                              stroke="#38bdf8"
                              strokeWidth={2}
                              dot={<CustomDot />}
                              activeDot={{ r: 5 }}
                            />

                            {/* Forecast projection */}
                            <Line
                              type="monotone"
                              dataKey="connector"
                              stroke="#22c55e"
                              strokeWidth={2.5}
                              dot={false}
                              activeDot={false}
                            />

                            {/* Terminal forecast point */}
                            {forecastPoint && (
                              <ReferenceDot
                                x={forecastPoint.date}
                                y={forecastPoint.price}
                                r={7}
                                fill="#ef4444"
                                stroke="#ffffff"
                                strokeWidth={1}
                                isFront
                              />
                            )}
                          </LineChart>
                        </ResponsiveContainer>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </section>

        {/* PORTFOLIO SECTION (unchanged) */}
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
          <h2 style={{ fontSize: "18px", fontWeight: 600, marginBottom: "10px" }}>
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
                <p style={{ fontSize: "16px", color: "#cbd5e1" }}>
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
                <p style={{ fontSize: "16px", color: "#cbd5e1" }}>
                  Run optimization to populate the weights table.
                </p>
              ) : (
                <table
                  style={{
                    width: "100%",
                    borderCollapse: "collapse",
                    fontSize: "16px",
                    background: "rgba(15,23,42,0.9)",
                    borderRadius: "12px",
                    overflow: "hidden",
                  }}
                >
                  <thead style={{ background: "rgba(15,23,42,0.95)" }}>
                    <tr style={{ color: "#f3f4f6" }}>
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


