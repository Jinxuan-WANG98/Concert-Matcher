const form = document.querySelector("#match-form");
const button = document.querySelector("#submit-button");
const statusPanel = document.querySelector("#status");
const fileWarning = document.querySelector("#file-warning");
const resultPanel = document.querySelector("#result-panel");
const resultsBody = document.querySelector("#results-body");
const resultSummary = document.querySelector("#result-summary");
const warningsBox = document.querySelector("#warnings");
const downloadLink = document.querySelector("#download-link");

const jobStorageKey = "concertMatchJobId";
const pollIntervalMs = 5000;
let activeJobId = null;
let pollTimer = null;
let pollInFlight = false;

const copy = {
  loading: "正在提交任务，请稍等。",
  matching: "匹配中",
  busy: "已有匹配任务在进行中。请保持页面打开，完成后会一次性显示最终结果。",
  start: "开始匹配",
  failed: "匹配失败，请稍后再试。",
  cancelled: "此前提交的任务已取消，未返回结果。你可以重新提交。",
  networkFailed: "状态查询暂时中断，正在自动重试；任务 ID 已保留，请勿重复提交。",
  jobInterrupted: "服务已重启或任务记录已过期；本次未返回结果，请重新提交。",
  empty: "暂时没有匹配到近期演出。可以换一个演出整理链接，或上传更清晰的图片。",
  fileMode: "你现在是直接打开 HTML 文件，这样只能看 UI，不能进行匹配。请先启动 Web App，然后打开 http://127.0.0.1:5050/ 。",
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
  if (text === "高") return "confidence confidence-high";
  if (text === "中") return "confidence confidence-medium";
  if (text === "低") return "confidence confidence-low";
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
      appendCell(row, (match.sample_songs || []).join("；"));

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

function displayErrorMessage(error, response) {
  const message = error && error.message ? error.message : "";
  if (response && response.status === 409) {
    return message || copy.busy;
  }
  if (message === "Failed to fetch" || message.includes("NetworkError")) {
    return copy.networkFailed;
  }
  return message || copy.failed;
}

function setSubmitting(isSubmitting) {
  button.disabled = isSubmitting;
  button.textContent = isSubmitting ? copy.matching : copy.start;
}

function rememberJob(jobId) {
  activeJobId = jobId;
  try {
    localStorage.setItem(jobStorageKey, jobId);
  } catch (error) {
    // Private browsing can reject storage; in-memory polling still works.
  }
}

function clearActiveJob() {
  activeJobId = null;
  if (pollTimer) {
    window.clearTimeout(pollTimer);
    pollTimer = null;
  }
  try {
    localStorage.removeItem(jobStorageKey);
  } catch (error) {
    // Ignore unavailable browser storage.
  }
}

function failActiveJob(message) {
  showStatus(message || copy.failed, true);
  clearActiveJob();
  setSubmitting(false);
}

async function cancelJob(jobId) {
  if (!jobId || window.location.protocol === "file:") return;
  try {
    await fetch(`/api/jobs/${jobId}/cancel`, {
      method: "POST",
      keepalive: true,
    });
  } catch (error) {
    // A page unload can interrupt the best-effort cancellation request.
  }
}

function scheduleJobPoll(jobId, delay = pollIntervalMs) {
  if (pollTimer) {
    window.clearTimeout(pollTimer);
  }
  pollTimer = window.setTimeout(() => pollJob(jobId), delay);
}

function progressMessage(data) {
  const message = data.message || copy.busy;
  const progress = Number.isFinite(data.progress) ? `（${data.progress}%）` : "";
  return `${message}${progress}`;
}

async function pollJob(jobId) {
  if (!jobId || activeJobId !== jobId) return;
  if (pollInFlight) {
    scheduleJobPoll(jobId);
    return;
  }

  pollInFlight = true;
  let shouldContinue = false;
  try {
    const response = await fetch(`/api/jobs/${jobId}`, { cache: "no-store" });
    const data = await parseJsonResponse(response);
    if (!response.ok) {
      if (response.status === 404) {
        failActiveJob(copy.jobInterrupted);
        return;
      }
      throw new Error(data.error || copy.failed);
    }

    if (data.state === "succeeded") {
      if (!data.result) {
        failActiveJob("任务已完成，但服务器没有返回最终结果。请重新提交。");
        return;
      }
      renderResults(data.result);
      hideStatus();
      clearActiveJob();
      setSubmitting(false);
      return;
    }

    if (data.state === "failed") {
      failActiveJob(data.error || copy.failed);
      return;
    }

    if (data.state === "cancelled") {
      failActiveJob(data.error || copy.cancelled);
      return;
    }

    showStatus(progressMessage(data));
    setSubmitting(true);
    shouldContinue = true;
  } catch (error) {
    showStatus(displayErrorMessage(error), true);
    if (activeJobId === jobId) {
      setSubmitting(true);
      shouldContinue = true;
    }
  } finally {
    pollInFlight = false;
    if (shouldContinue && activeJobId === jobId) {
      scheduleJobPoll(jobId);
    }
  }
}

async function resumeStoredJob() {
  if (window.location.protocol === "file:") return;

  let storedJobId = null;
  try {
    storedJobId = localStorage.getItem(jobStorageKey);
  } catch (error) {
    storedJobId = null;
  }
  if (storedJobId) {
    await cancelJob(storedJobId);
    clearActiveJob();
    showStatus(copy.cancelled, true);
  }
  setSubmitting(false);
}

window.addEventListener("pagehide", () => {
  if (activeJobId) {
    void cancelJob(activeJobId);
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (window.location.protocol === "file:") {
    showStatus(copy.fileMode, true);
    return;
  }
  if (activeJobId) {
    showStatus(copy.busy);
    return;
  }

  resultPanel.classList.add("hidden");
  showStatus(copy.loading);
  setSubmitting(true);

  let response;
  try {
    response = await fetch("/api/match", {
      method: "POST",
      body: new FormData(form),
    });
    const data = await parseJsonResponse(response);
    if (!response.ok) {
      throw new Error(data.error || copy.failed);
    }
    if (!data.job_id) {
      throw new Error("服务器没有返回任务 ID，请稍后重试。");
    }

    rememberJob(data.job_id);
    showStatus(copy.busy);
    await pollJob(data.job_id);
  } catch (error) {
    showStatus(displayErrorMessage(error, response), true);
    if (!activeJobId) {
      setSubmitting(false);
    }
  }
});

resumeStoredJob();
