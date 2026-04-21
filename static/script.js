// Global variables to store charts
let rulProgressionChart = null;
let healthDistributionChart = null;
let rulHistogramChart = null;
let urgencyChart = null;

// ==================== [NEW] STREAMING STATE ====================
// Accumulates data as SSE events arrive
let streamedRulMinutes = [];
let streamedIndices = [];
let streamedHealthStates = [];
let activeEventSource = null; // reference so we can close it if needed

// ==================== FILE INPUT HANDLERS ====================

document
  .getElementById("misalignTempFile")
  .addEventListener("change", function (e) {
    const fileName = e.target.files[0]?.name || "No file chosen";
    document.getElementById("misalignTempFileName").textContent = fileName;
  });

document
  .getElementById("misalignVibFile")
  .addEventListener("change", function (e) {
    const fileName = e.target.files[0]?.name || "No file chosen";
    document.getElementById("misalignVibFileName").textContent = fileName;
  });

document
  .getElementById("bpfiTempFile")
  .addEventListener("change", function (e) {
    const fileName = e.target.files[0]?.name || "No file chosen";
    document.getElementById("bpfiTempFileName").textContent = fileName;
  });

document.getElementById("bpfiVibFile").addEventListener("change", function (e) {
  const fileName = e.target.files[0]?.name || "No file chosen";
  document.getElementById("bpfiVibFileName").textContent = fileName;
});

// ==================== FORM SUBMISSION HANDLERS ====================

document
  .getElementById("misalignForm")
  .addEventListener("submit", async function (e) {
    e.preventDefault();
    await handleFormSubmission("misalign");
  });

document
  .getElementById("bpfiForm")
  .addEventListener("submit", async function (e) {
    e.preventDefault();
    await handleFormSubmission("bpfi");
  });

// ==================== [MODIFIED] FORM SUBMISSION → STREAMING ====================

async function handleFormSubmission(modelType) {
  const tempFileId = `${modelType}TempFile`;
  const vibFileId = `${modelType}VibFile`;
  const btnId = `${modelType}PredictBtn`;
  const btnTextId = `${modelType}BtnText`;
  const btnLoaderId = `${modelType}BtnLoader`;

  const tempFile = document.getElementById(tempFileId).files[0];
  const vibFile = document.getElementById(vibFileId).files[0];

  if (!tempFile || !vibFile) {
    showAlert("error", "Please select both files");
    return;
  }

  // Show loading state
  const predictBtn = document.getElementById(btnId);
  const btnText = document.getElementById(btnTextId);
  const btnLoader = document.getElementById(btnLoaderId);

  predictBtn.disabled = true;
  btnText.style.display = "none";
  btnLoader.style.display = "inline-block";

  // Close any existing stream
  if (activeEventSource) {
    activeEventSource.close();
    activeEventSource = null;
  }

  // Reset streaming accumulators
  streamedRulMinutes = [];
  streamedIndices = [];
  streamedHealthStates = [];

  // POST files to /predict_stream via fetch, then open the SSE response
  const formData = new FormData();
  formData.append("temp_file", tempFile);
  formData.append("vib_file", vibFile);
  formData.append("model_type", modelType);

  try {
    // Use fetch to POST files; the response body IS the SSE stream
    const response = await fetch("/predict_stream", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Server returned ${response.status}`);
    }

    // Show sections early so charts appear immediately
    document.getElementById("statsSection").style.display = "grid";
    document.getElementById("chartsSection").style.display = "grid";

    // Show model badge
    const modelInfo = document.getElementById("modelInfo");
    const modelBadge = document.getElementById("modelBadge");
    modelInfo.style.display = "block";
    modelBadge.textContent = `RUL Model: ${modelType.toUpperCase()}`;
    modelBadge.className = `model-badge model-badge-${modelType}`;

    // Initialize empty charts
    initEmptyCharts();

    // Read SSE stream manually from the fetch response body
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // SSE lines end with \n\n
      const parts = buffer.split("\n\n");
      buffer = parts.pop(); // keep incomplete last chunk

      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith("data:")) continue;

        const jsonStr = line.slice(5).trim();
        let payload;
        try {
          payload = JSON.parse(jsonStr);
        } catch {
          continue;
        }

        handleSSEEvent(payload, modelType, predictBtn, btnText, btnLoader);
      }
    }
  } catch (error) {
    console.error("Streaming error:", error);
    showAlert("error", "An error occurred during prediction");
    predictBtn.disabled = false;
    btnText.style.display = "inline";
    btnLoader.style.display = "none";
  }
}

// ==================== [NEW] SSE EVENT HANDLER ====================

function handleSSEEvent(payload, modelType, predictBtn, btnText, btnLoader) {
  switch (payload.event) {
    case "start":
      showAlert(
        "success",
        `Processing ${payload.total} sequences — streaming results…`,
      );
      break;

    case "prediction": {
      // Append new data point
      streamedIndices.push(payload.index);
      streamedRulMinutes.push(payload.rul_minutes);
      streamedHealthStates.push(payload.health_state);

      // Live-update the RUL progression chart only
      updateRULProgressionChart();

      // Live-update summary stat cards
      updateLiveStatCards();
      break;
    }

    case "done": {
      // Stream finished — render full statistics + classification
      const { statistics, classifications } = payload;

      // Final chart renders with complete data
      createHealthDistributionChart(statistics);
      createRULHistogramChart(streamedRulMinutes);
      createUrgencyChart(statistics);

      // Full stats table
      populateStatisticsTable(statistics);

      // Classification cards
      if (classifications && Object.keys(classifications).length > 0) {
        displayClassificationCards(classifications);
      } else {
        document.getElementById("classificationSection").style.display = "none";
      }

      // Alerts
      if (statistics.critical_count > 0) {
        showAlert(
          "warning",
          `🚨 CRITICAL: ${statistics.critical_count} sequences need immediate maintenance!`,
        );
      } else if (statistics.severe_count > 0) {
        showAlert(
          "warning",
          `⚠️ WARNING: ${statistics.severe_count} sequences in severe condition. Plan maintenance soon.`,
        );
      } else {
        showAlert(
          "success",
          `✅ Analysis complete — ${statistics.total_sequences} sequences processed.`,
        );
      }

      // Re-enable button
      predictBtn.disabled = false;
      btnText.style.display = "inline";
      btnLoader.style.display = "none";
      break;
    }

    case "error":
      showAlert("error", payload.message || "Prediction failed");
      predictBtn.disabled = false;
      btnText.style.display = "inline";
      btnLoader.style.display = "none";
      break;
  }
}

// ==================== [NEW] LIVE STAT CARD UPDATER ====================

function updateLiveStatCards() {
  if (streamedRulMinutes.length === 0) return;

  const mean =
    streamedRulMinutes.reduce((a, b) => a + b, 0) / streamedRulMinutes.length;

  let healthy = 0,
    warning = 0,
    severe = 0,
    critical = 0;
  for (const s of streamedHealthStates) {
    if (s === "Healthy") healthy++;
    else if (s === "Warning") warning++;
    else if (s === "Severe") severe++;
    else if (s === "Critical") critical++;
  }

  document.getElementById("totalSequences").textContent =
    streamedRulMinutes.length;
  document.getElementById("meanRUL").textContent = `${mean.toFixed(1)} min`;
  document.getElementById("warningCount").textContent = warning;
  document.getElementById("severeCount").textContent = severe;
  document.getElementById("criticalCount").textContent = critical;
}

// ==================== [NEW] INIT EMPTY CHARTS ====================

function initEmptyCharts() {
  // Destroy existing charts
  [
    rulProgressionChart,
    healthDistributionChart,
    rulHistogramChart,
    urgencyChart,
  ].forEach((c) => {
    if (c) c.destroy();
  });
  rulProgressionChart = null;
  healthDistributionChart = null;
  rulHistogramChart = null;
  urgencyChart = null;

  // Create an empty RUL progression chart so it appears instantly
  const ctx = document.getElementById("rulProgressionChart").getContext("2d");
  const canvas = document.getElementById("rulProgressionChart");
  canvas.style.width = "800px";

  rulProgressionChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Predicted RUL (minutes)",
          data: [],
          borderColor: "#3b82f6",
          backgroundColor: "rgba(59, 130, 246, 0.1)",
          borderWidth: 2,
          fill: true,
          tension: 0.4,
          pointRadius: 3,
          pointHoverRadius: 6,
        },
      ],
    },
    options: rulProgressionOptions(),
  });
}

// ==================== [NEW] LIVE CHART UPDATE ====================

function updateRULProgressionChart() {
  if (!rulProgressionChart) return;

  // Dynamically widen canvas as data grows
  const canvas = document.getElementById("rulProgressionChart");
  const minWidth = Math.max(800, streamedIndices.length * 15);
  canvas.style.width = `${minWidth}px`;

  rulProgressionChart.data.labels = [...streamedIndices];
  rulProgressionChart.data.datasets[0].data = [...streamedRulMinutes];
  rulProgressionChart.update("none"); // "none" = no animation per update (smoother)
}

// ==================== CHART OPTIONS HELPER ====================

function rulProgressionOptions() {
  return {
    responsive: true,
    maintainAspectRatio: true,
    aspectRatio: 3,
    animation: { duration: 300 },
    plugins: {
      legend: { display: true, position: "top" },
      tooltip: {
        callbacks: {
          label: (ctx) => {
            const rul = ctx.parsed.y.toFixed(2);
            const hours = (ctx.parsed.y / 60).toFixed(2);
            return `RUL: ${rul} min (${hours} hrs)`;
          },
        },
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: "RUL (minutes)",
          font: { size: 14, weight: "bold" },
        },
        ticks: { callback: (v) => v.toFixed(0) },
      },
      x: {
        title: {
          display: true,
          text: "Sequence Index",
          font: { size: 14, weight: "bold" },
        },
        ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 20 },
      },
    },
  };
}

// ==================== POPULATE STATS TABLE ====================

function populateStatisticsTable(statistics) {
  document.getElementById("detailMeanRUL").textContent =
    `${(statistics.mean_rul_minutes || 0).toFixed(2)} min`;
  document.getElementById("detailMedianRUL").textContent =
    `${(statistics.median_rul_minutes || 0).toFixed(2)} min`;
  document.getElementById("minRUL").textContent =
    `${(statistics.min_rul_minutes || 0).toFixed(2)} min`;
  document.getElementById("maxRUL").textContent =
    `${(statistics.max_rul_minutes || 0).toFixed(2)} min`;
  document.getElementById("rangeRUL").textContent =
    `${(statistics.range_rul_minutes || 0).toFixed(2)} min`;
  document.getElementById("stdRUL").textContent =
    `${(statistics.std_rul_minutes || 0).toFixed(2)} min`;
  document.getElementById("varianceRUL").textContent =
    `${(statistics.variance_rul_minutes || 0).toFixed(2)}`;
  document.getElementById("q1RUL").textContent =
    `${(statistics.q1_rul_minutes || 0).toFixed(2)} min`;
  document.getElementById("q3RUL").textContent =
    `${(statistics.q3_rul_minutes || 0).toFixed(2)} min`;
  document.getElementById("iqrRUL").textContent =
    `${(statistics.iqr_rul_minutes || 0).toFixed(2)} min`;
  document.getElementById("healthyCount").textContent =
    statistics.healthy_count || 0;
}

// ==================== DISPLAY RESULTS (kept for compatibility) ====================

function displayResults(data) {
  console.log("Displaying results:", data);

  const { predictions, statistics, classifications } = data;

  if (!predictions || !statistics) {
    showAlert("error", "Invalid response data");
    return;
  }

  const modelInfo = document.getElementById("modelInfo");
  const modelBadge = document.getElementById("modelBadge");
  modelInfo.style.display = "block";
  modelBadge.textContent = `RUL Model: ${(statistics.model_type || "unknown").toUpperCase()}`;
  modelBadge.className = `model-badge model-badge-${statistics.model_type || "misalign"}`;

  if (classifications && Object.keys(classifications).length > 0) {
    displayClassificationCards(classifications);
  } else {
    document.getElementById("classificationSection").style.display = "none";
  }

  document.getElementById("statsSection").style.display = "grid";
  document.getElementById("chartsSection").style.display = "grid";

  document.getElementById("totalSequences").textContent =
    statistics.total_sequences || 0;
  document.getElementById("meanRUL").textContent =
    `${(statistics.mean_rul_minutes || 0).toFixed(1)} min`;
  document.getElementById("warningCount").textContent =
    statistics.warning_count || 0;
  document.getElementById("severeCount").textContent =
    statistics.severe_count || 0;
  document.getElementById("criticalCount").textContent =
    statistics.critical_count || 0;

  populateStatisticsTable(statistics);
  createRULProgressionChart(predictions);
  createHealthDistributionChart(statistics);
  createRULHistogramChart(predictions.rul_minutes || []);
  createUrgencyChart(statistics);

  if (statistics.critical_count > 0) {
    showAlert(
      "warning",
      `🚨 CRITICAL: ${statistics.critical_count} sequences need immediate maintenance!`,
    );
  } else if (statistics.severe_count > 0) {
    showAlert(
      "warning",
      `⚠️ WARNING: ${statistics.severe_count} sequences in severe condition. Plan maintenance soon.`,
    );
  }
}

// ==================== DISPLAY CLASSIFICATION CARDS ====================

function displayClassificationCards(classifications) {
  const classificationSection = document.getElementById(
    "classificationSection",
  );
  const cardsContainer = document.getElementById("classificationCards");

  if (!classifications || Object.keys(classifications).length === 0) {
    classificationSection.style.display = "none";
    return;
  }

  classificationSection.style.display = "block";
  cardsContainer.innerHTML = "";

  if (classifications.ensemble) {
    displayEnsembleCard(classifications.ensemble, cardsContainer);
  }
}

function displayEnsembleCard(result, container) {
  let severityClass = "normal";
  let severityIcon = "✅";

  if (result.fault_type !== "Normal") {
    if (result.maintenance_recommendation.includes("CRITICAL")) {
      severityClass = "critical";
      severityIcon = "🚨";
    } else if (result.maintenance_recommendation.includes("WARNING")) {
      severityClass = "warning";
      severityIcon = "⚠️";
    } else {
      severityClass = "caution";
      severityIcon = "⚡";
    }
  }

  const card = document.createElement("div");
  card.className = `classification-card ${severityClass}`;

  let distributionHTML = "";
  if (
    result.class_distribution &&
    Object.keys(result.class_distribution).length > 0
  ) {
    distributionHTML =
      '<div class="class-distribution"><h4>📊 Prediction Distribution:</h4><ul>';
    for (const [className, data] of Object.entries(result.class_distribution)) {
      distributionHTML += `<li><strong>${className}:</strong> ${data.count} segments (${data.percentage.toFixed(1)}%)</li>`;
    }
    distributionHTML += "</ul></div>";
  }

  card.innerHTML = `
    <div class="classification-header">
      <h3>🤖 ${result.model_type || "Ensemble Model"}</h3>
      <span class="confidence-badge">
        ${(result.confidence * 100).toFixed(1)}% Confidence
      </span>
    </div>
    <div class="classification-body">
      <div class="classification-item">
        <span class="item-label">Predicted Class</span>
        <span class="item-value class-name">${result.predicted_class}</span>
      </div>
      <div class="classification-item">
        <span class="item-label">Fault Type</span>
        <span class="item-value fault-type">${result.fault_description}</span>
      </div>
      <div class="classification-item">
        <span class="item-label">Severity</span>
        <span class="item-value severity-badge severity-${severityClass}">
          ${severityIcon} ${result.severity}
        </span>
      </div>
      <div class="classification-item full-width">
        <span class="item-label">Maintenance Action</span>
        <div class="maintenance-box">
          <p>${result.maintenance_recommendation}</p>
        </div>
      </div>
      ${distributionHTML}
      <div class="classification-stats">
        <div class="stat-small">
          <span class="stat-label">Total Segments Analyzed</span>
          <span class="stat-value">${result.total_segments_analyzed}</span>
        </div>
      </div>
    </div>
  `;

  container.appendChild(card);
}

// ==================== CREATE CHARTS ====================

function createRULProgressionChart(predictions) {
  const ctx = document.getElementById("rulProgressionChart").getContext("2d");

  if (rulProgressionChart) {
    rulProgressionChart.destroy();
  }

  const data = predictions.rul_minutes || [];
  const indices = predictions.sequence_indices || [];

  if (data.length === 0) return;

  const canvas = document.getElementById("rulProgressionChart");
  const minWidth = Math.max(800, indices.length * 15);
  canvas.style.width = `${minWidth}px`;

  rulProgressionChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: indices,
      datasets: [
        {
          label: "Predicted RUL (minutes)",
          data: data,
          borderColor: "#3b82f6",
          backgroundColor: "rgba(59, 130, 246, 0.1)",
          borderWidth: 2,
          fill: true,
          tension: 0.4,
          pointRadius: 3,
          pointHoverRadius: 6,
        },
      ],
    },
    options: rulProgressionOptions(),
  });
}

function createHealthDistributionChart(statistics) {
  const ctx = document
    .getElementById("healthDistributionChart")
    .getContext("2d");

  if (healthDistributionChart) healthDistributionChart.destroy();

  healthDistributionChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Healthy", "Warning", "Severe", "Critical"],
      datasets: [
        {
          data: [
            statistics.healthy_count || 0,
            statistics.warning_count || 0,
            statistics.severe_count || 0,
            statistics.critical_count || 0,
          ],
          backgroundColor: ["#22c55e", "#f59e0b", "#ff6b35", "#ef4444"],
          borderWidth: 2,
          borderColor: "#ffffff",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: {
          position: "bottom",
          labels: {
            padding: 15,
            font: { size: 12 },
            generateLabels: (chart) => {
              const d = chart.data;
              if (d.labels.length && d.datasets.length) {
                return d.labels.map((label, i) => {
                  const value = d.datasets[0].data[i];
                  const total = d.datasets[0].data.reduce((a, b) => a + b, 0);
                  const percentage =
                    total > 0 ? ((value / total) * 100).toFixed(1) : 0;
                  return {
                    text: `${label}: ${value} (${percentage}%)`,
                    fillStyle: d.datasets[0].backgroundColor[i],
                    hidden: false,
                    index: i,
                  };
                });
              }
              return [];
            },
          },
        },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const total = statistics.total_sequences || 1;
              const percentage = ((ctx.parsed / total) * 100).toFixed(1);
              return `${ctx.label}: ${ctx.parsed} (${percentage}%)`;
            },
          },
        },
      },
    },
  });
}

function createRULHistogramChart(rulData) {
  const ctx = document.getElementById("rulHistogramChart").getContext("2d");

  if (rulHistogramChart) rulHistogramChart.destroy();

  if (!rulData || rulData.length === 0) return;

  const bins = 20;
  const min = Math.min(...rulData);
  const max = Math.max(...rulData);
  const binSize = (max - min) / bins;

  const histogram = Array(bins).fill(0);
  const labels = [];

  for (let i = 0; i < bins; i++) {
    const binStart = min + i * binSize;
    const binEnd = binStart + binSize;
    labels.push(`${binStart.toFixed(0)}-${binEnd.toFixed(0)}`);
    for (const value of rulData) {
      if (
        value >= binStart &&
        (value < binEnd || (i === bins - 1 && value === binEnd))
      ) {
        histogram[i]++;
      }
    }
  }

  rulHistogramChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Frequency",
          data: histogram,
          backgroundColor: "#3b82f6",
          borderColor: "#2563eb",
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const total = rulData.length;
              const percentage =
                total > 0 ? ((ctx.parsed.y / total) * 100).toFixed(1) : 0;
              return `Count: ${ctx.parsed.y} (${percentage}%)`;
            },
          },
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: "Frequency",
            font: { size: 14, weight: "bold" },
          },
          ticks: { precision: 0 },
        },
        x: {
          title: {
            display: true,
            text: "RUL Range (minutes)",
            font: { size: 14, weight: "bold" },
          },
          ticks: { maxRotation: 45, minRotation: 45 },
        },
      },
    },
  });
}

function createUrgencyChart(statistics) {
  const ctx = document.getElementById("urgencyChart").getContext("2d");

  if (urgencyChart) urgencyChart.destroy();

  const total = statistics.total_sequences || 1;

  urgencyChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Low", "Medium", "High", "Critical"],
      datasets: [
        {
          label: "Count",
          data: [
            statistics.healthy_count || 0,
            statistics.warning_count || 0,
            statistics.severe_count || 0,
            statistics.critical_count || 0,
          ],
          backgroundColor: ["#22c55e", "#f59e0b", "#ff6b35", "#ef4444"],
          borderColor: ["#16a34a", "#d97706", "#e55b2d", "#dc2626"],
          borderWidth: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const value = ctx.parsed.y;
              const percentage = ((value / total) * 100).toFixed(1);
              return `Count: ${value} (${percentage}%)`;
            },
          },
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: "Number of Sequences",
            font: { size: 14, weight: "bold" },
          },
          ticks: { precision: 0 },
        },
        x: {
          title: {
            display: true,
            text: "Maintenance Urgency",
            font: { size: 14, weight: "bold" },
          },
        },
      },
    },
  });
}

// ==================== [NEW] SHOW / HIDE DETAILS TOGGLE ====================

function toggleDetails() {
  const detailsSection = document.getElementById("detailsToggleSection");
  const toggleBtn = document.getElementById("toggleDetailsBtn");

  if (!detailsSection || !toggleBtn) return;

  const isHidden =
    detailsSection.style.display === "none" ||
    detailsSection.style.display === "";

  if (isHidden) {
    detailsSection.style.display = "block";
    toggleBtn.textContent = "🔼 Hide Details";
  } else {
    detailsSection.style.display = "none";
    toggleBtn.textContent = "🔽 Show Details";
  }
}

// ==================== ALERT SYSTEM ====================

function showAlert(type, message) {
  const alertBox = document.getElementById("alertBox");
  const alertIcon = document.getElementById("alertIcon");
  const alertMessage = document.getElementById("alertMessage");

  alertBox.classList.remove("alert-success", "alert-error", "alert-warning");

  if (type === "success") {
    alertIcon.textContent = "✓";
    alertBox.classList.add("alert-success");
  } else if (type === "error") {
    alertIcon.textContent = "✗";
    alertBox.classList.add("alert-error");
  } else if (type === "warning") {
    alertIcon.textContent = "⚠️";
    alertBox.classList.add("alert-warning");
  }

  alertMessage.textContent = message;
  alertBox.style.display = "flex";

  setTimeout(() => {
    closeAlert();
  }, 5000);
}

function closeAlert() {
  document.getElementById("alertBox").style.display = "none";
}
