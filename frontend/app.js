// Empty base = same-origin (works when served by FastAPI at :8000).
// Override here if hosting the static files elsewhere.
const API_BASE_URL = window.API_BASE_URL || "";

const els = {
    tickerInput: document.getElementById("ticker-input"),
    predictBtn: document.getElementById("predict-btn"),
    status: document.getElementById("status-line"),
    arrow: document.getElementById("prediction-arrow"),
    direction: document.getElementById("prediction-direction"),
    confidence: document.getElementById("prediction-confidence"),
    model: document.getElementById("prediction-model"),
    barPositive: document.getElementById("bar-positive"),
    barNeutral: document.getElementById("bar-neutral"),
    barNegative: document.getElementById("bar-negative"),
    valPositive: document.getElementById("val-positive"),
    valNeutral: document.getElementById("val-neutral"),
    valNegative: document.getElementById("val-negative"),
    sentimentUpdated: document.getElementById("sentiment-updated"),
    historyBody: document.getElementById("history-body"),
    tickerHint: document.getElementById("ticker-hint"),
};

const history = [];

function setStatus(message, kind = "info") {
    els.status.textContent = message || "";
    els.status.dataset.kind = kind;
}

function setLoading(isLoading) {
    els.predictBtn.disabled = isLoading;
    els.predictBtn.textContent = isLoading ? "Loading…" : "Predict";
}

async function predict(ticker) {
    // window_data omitted → API loads the latest features for this ticker from disk.
    const res = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker }),
    });
    if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `Predict failed (${res.status})`);
    }
    return res.json();
}

async function getSentiment(ticker) {
    const res = await fetch(`${API_BASE_URL}/sentiment/${encodeURIComponent(ticker)}`);
    if (!res.ok) throw new Error(`Sentiment failed (${res.status})`);
    return res.json();
}

async function loadAvailableTickers() {
    try {
        const res = await fetch(`${API_BASE_URL}/tickers`);
        if (!res.ok) return;
        const data = await res.json();
        if (els.tickerHint && data.tickers && data.tickers.length) {
            els.tickerHint.textContent = `Available: ${data.tickers.join(", ")}`;
        }
    } catch (err) {
        console.warn("Could not load ticker list", err);
    }
}

function renderPrediction(p) {
    const dirUp = p.direction === "up";
    els.arrow.textContent = dirUp ? "▲" : "▼";
    els.arrow.className = `arrow ${dirUp ? "up" : "down"}`;
    els.direction.textContent = dirUp ? "Predicted UP" : "Predicted DOWN";
    els.direction.className = `direction ${dirUp ? "up" : "down"}`;
    els.confidence.textContent = `${(p.confidence * 100).toFixed(1)}%`;
    els.model.textContent = (p.model || "").toUpperCase();
}

function renderSentiment(s) {
    const pct = (v) => `${(Math.max(0, Math.min(1, v)) * 100).toFixed(0)}%`;
    els.barPositive.style.width = pct(s.positive);
    els.barNeutral.style.width = pct(s.neutral);
    els.barNegative.style.width = pct(s.negative);
    els.valPositive.textContent = pct(s.positive);
    els.valNeutral.textContent = pct(s.neutral);
    els.valNegative.textContent = pct(s.negative);
    els.sentimentUpdated.textContent = `Last updated: ${s.last_updated}`;
}

function renderHistory() {
    if (!history.length) {
        els.historyBody.innerHTML = '<tr><td colspan="5" class="muted">No predictions yet.</td></tr>';
        return;
    }
    els.historyBody.innerHTML = history
        .slice(-8)
        .reverse()
        .map(
            (h) => `
            <tr>
                <td>${h.time}</td>
                <td>${h.ticker}</td>
                <td class="${h.direction}">${h.direction.toUpperCase()}</td>
                <td>${(h.confidence * 100).toFixed(1)}%</td>
                <td>${(h.model || "").toUpperCase()}</td>
            </tr>`
        )
        .join("");
}

async function onPredictClick() {
    const ticker = (els.tickerInput.value || "").trim().toUpperCase();
    if (!ticker) {
        setStatus("Enter a ticker first.", "error");
        return;
    }

    setLoading(true);
    setStatus(`Fetching prediction & sentiment for ${ticker}…`);

    try {
        const [pred, sent] = await Promise.all([predict(ticker), getSentiment(ticker)]);
        renderPrediction(pred);
        renderSentiment(sent);
        history.push({
            time: new Date().toLocaleTimeString(),
            ticker: pred.ticker,
            direction: pred.direction,
            confidence: pred.confidence,
            model: pred.model,
        });
        renderHistory();
        setStatus(`Done (${ticker}).`, "success");
    } catch (err) {
        console.error(err);
        setStatus(`Error: ${err.message}`, "error");
    } finally {
        setLoading(false);
    }
}

els.predictBtn.addEventListener("click", onPredictClick);
els.tickerInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") onPredictClick();
});

loadAvailableTickers();
