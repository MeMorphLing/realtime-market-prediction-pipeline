const API_BASE_URL = "http://localhost:8000";
const WINDOW_SIZE = 30;
const N_FEATURES = 9;

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

function buildDummyWindow() {
    const window = [];
    for (let t = 0; t < WINDOW_SIZE; t++) {
        const row = [];
        for (let f = 0; f < N_FEATURES; f++) row.push(0);
        window.push(row);
    }
    return window;
}

async function predict(ticker) {
    const url = `${API_BASE_URL}/predict`;
    const body = { ticker, window_data: buildDummyWindow() };
    const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`Predict failed (${res.status})`);
    return res.json();
}

async function getSentiment(ticker) {
    const res = await fetch(`${API_BASE_URL}/sentiment/${encodeURIComponent(ticker)}`);
    if (!res.ok) throw new Error(`Sentiment failed (${res.status})`);
    return res.json();
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
