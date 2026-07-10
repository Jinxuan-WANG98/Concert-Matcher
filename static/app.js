const form = document.querySelector("#match-form");
const button = document.querySelector("#submit-button");
const statusPanel = document.querySelector("#status");
const fileWarning = document.querySelector("#file-warning");
const resultPanel = document.querySelector("#result-panel");
const resultsBody = document.querySelector("#results-body");
const resultSummary = document.querySelector("#result-summary");
const warningsBox = document.querySelector("#warnings");
const downloadLink = document.querySelector("#download-link");

const copy = {
  loading: "\u6b63\u5728\u5e2e\u4f60\u8bfb\u53d6\u6b4c\u5355\u3001\u8bc6\u522b\u56fe\u7247\u548c\u5339\u914d\u6b4c\u624b\uff0c\u8bf7\u7a0d\u7b49\u3002",
  matching: "\u5339\u914d\u4e2d",
  start: "\u5f00\u59cb\u5339\u914d",
  failed: "\u5339\u914d\u5931\u8d25\uff0c\u8bf7\u7a0d\u540e\u518d\u8bd5\u3002",
  networkFailed: "\u8fde\u63a5\u4e2d\u65ad\uff1a\u672c\u5730\u670d\u52a1\u53ef\u80fd\u521a\u91cd\u542f\uff0c\u6216\u8bf7\u6c42\u65f6\u95f4\u592a\u957f\u88ab\u6d4f\u89c8\u5668\u4e2d\u65ad\u3002\u8bf7\u5237\u65b0\u9875\u9762\u540e\u91cd\u8bd5\u3002",
  empty: "\u6682\u65f6\u6ca1\u6709\u5339\u914d\u5230\u8fd1\u671f\u6f14\u51fa\u3002\u53ef\u4ee5\u6362\u4e00\u4e2a\u6f14\u51fa\u6574\u7406\u94fe\u63a5\uff0c\u6216\u4e0a\u4f20\u66f4\u6e05\u6670\u7684\u56fe\u7247\u3002",
  fileMode: "\u4f60\u73b0\u5728\u662f\u76f4\u63a5\u6253\u5f00 HTML \u6587\u4ef6\uff0c\u8fd9\u6837\u53ea\u80fd\u770b UI\uff0c\u4e0d\u80fd\u8fdb\u884c\u5339\u914d\u3002\u8bf7\u5148\u542f\u52a8 Web App\uff0c\u7136\u540e\u6253\u5f00 http://127.0.0.1:5050/ \u3002",
};

if (window.location.protocol === "file:") {
  fileWarning.textContent = copy.fileMode;
  fileWarning.classList.remove("hidden");
}

function showStatus(message, isError = false) {
  statusPanel.textContent = message;
  statusPanel.classList.remove("hidden");
  statusPanel.style.color = isError ? "#a13d5e" : "";
}

function hideStatus() {
  statusPanel.classList.add("hidden");
  statusPanel.textContent = "";
}

function renderWarnings(warnings) {
  if (!warnings || warnings.length === 0) {
    warningsBox.classList.add("hidden");
    warningsBox.textContent = "";
    return;
  }
  warningsBox.textContent = warnings.join(" ");
  warningsBox.classList.remove("hidden");
}

function appendCell(row, value) {
  const cell = document.createElement("td");
  cell.textContent = value == null ? "" : String(value);
  row.appendChild(cell);
}

function confidenceClass(value) {
  const text = String(value || "").trim();
  if (text === "\u9ad8") return "confidence confidence-high";
  if (text === "\u4e2d") return "confidence confidence-medium";
  if (text === "\u4f4e") return "confidence confidence-low";
  return "confidence";
}

function renderResults(data) {
  resultsBody.innerHTML = "";
  const matches = data.matches || [];
  const eventCount = Number.isInteger(data.event_count) ? data.event_count : null;
  resultSummary.textContent = eventCount == null
    ? `匹配到 ${matches.length} 位歌手`
    : `识别到 ${eventCount} 场演出，匹配到 ${matches.length} 位歌手`;

  if (matches.length === 0) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.className = "empty-row";
    cell.colSpan = 7;
    cell.textContent = copy.empty;
    row.appendChild(cell);
    resultsBody.appendChild(row);
  } else {
    for (const match of matches) {
      const row = document.createElement("tr");
      appendCell(row, match.index);
      appendCell(row, match.date);
      appendCell(row, match.artist);
      appendCell(row, match.venue || "");
      appendCell(row, match.playlist_song_count);
      appendCell(row, (match.sample_songs || []).join("\uff1b"));

      const confidenceCell = document.createElement("td");
      const confidence = document.createElement("span");
      confidence.className = confidenceClass(match.confidence);
      confidence.textContent = match.confidence || "";
      confidenceCell.appendChild(confidence);
      row.appendChild(confidenceCell);
      resultsBody.appendChild(row);
    }
  }

  renderWarnings(data.warnings);
  if (data.download_url) {
    downloadLink.href = data.download_url;
    downloadLink.classList.remove("hidden");
  } else {
    downloadLink.classList.add("hidden");
  }
  resultPanel.classList.remove("hidden");
}

async function parseJsonResponse(response) {
  const text = await response.text();
  try {
    return JSON.parse(text);
  } catch (error) {
    return { error: text || copy.failed };
  }
}

function displayErrorMessage(error) {
  const message = error && error.message ? error.message : "";
  if (message === "Failed to fetch" || message.includes("NetworkError")) {
    return copy.networkFailed;
  }
  return message || copy.failed;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (window.location.protocol === "file:") {
    showStatus(copy.fileMode, true);
    return;
  }
  resultPanel.classList.add("hidden");
  showStatus(copy.loading);
  button.disabled = true;
  button.textContent = copy.matching;

  try {
    const response = await fetch("/api/match", {
      method: "POST",
      body: new FormData(form),
    });
    const data = await parseJsonResponse(response);
    if (!response.ok) {
      throw new Error(data.error || copy.failed);
    }
    hideStatus();
    renderResults(data);
  } catch (error) {
    showStatus(displayErrorMessage(error), true);
  } finally {
    button.disabled = false;
    button.textContent = copy.start;
  }
});
