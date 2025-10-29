// Global variables to store charts
let rulProgressionChart = null;
let healthDistributionChart = null;
let rulHistogramChart = null;
let urgencyChart = null;

// ==================== FILE INPUT HANDLERS ====================

// Misalignment file inputs
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

// BPFI file inputs
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

// Misalignment form submission
document
  .getElementById("misalignForm")
  .addEventListener("submit", async function (e) {
    e.preventDefault();
    await handleFormSubmission("misalign");
  });

// BPFI form submission
document
  .getElementById("bpfiForm")
  .addEventListener("submit", async function (e) {
    e.preventDefault();
    await handleFormSubmission("bpfi");
  });

// Generic form submission handler
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

  // Prepare form data
  const formData = new FormData();
  formData.append("temp_file", tempFile);
  formData.append("vib_file", vibFile);
  formData.append("model_type", modelType);

  try {
    // Send request to Flask backend
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (data.success) {
      showAlert(
        "success",
        `Predictions generated successfully using ${modelType} model!`
      );
      displayResults(data);
    } else {
      showAlert("error", data.message || data.error || "Prediction failed");
    }
  } catch (error) {
    console.error("Error:", error);
    showAlert("error", "An error occurred during prediction");
  } finally {
    // Reset button state
    predictBtn.disabled = false;
    btnText.style.display = "inline";
    btnLoader.style.display = "none";
  }
}

// ==================== DISPLAY RESULTS ====================

function displayResults(data) {
  console.log("Displaying results:", data);

  const { predictions, statistics } = data;

  // Validate data
  if (!predictions || !statistics) {
    showAlert("error", "Invalid response data");
    return;
  }

  // Show model info
  const modelInfo = document.getElementById("modelInfo");
  const modelBadge = document.getElementById("modelBadge");
  modelInfo.style.display = "block";
  modelBadge.textContent = `Model: ${(
    statistics.model_type || "unknown"
  ).toUpperCase()}`;
  modelBadge.className = `model-badge model-badge-${
    statistics.model_type || "misalign"
  }`;

  // Show sections
  document.getElementById("statsSection").style.display = "grid";
  document.getElementById("chartsSection").style.display = "grid";

  // Update statistics cards with safe defaults
  document.getElementById("totalSequences").textContent =
    statistics.total_sequences || 0;
  document.getElementById("meanRUL").textContent = `${(
    statistics.mean_rul_minutes || 0
  ).toFixed(1)} min`;
  document.getElementById("warningCount").textContent =
    statistics.warning_count || 0;
  document.getElementById("severeCount").textContent =
    statistics.severe_count || 0;
  document.getElementById("criticalCount").textContent =
    statistics.critical_count || 0;

  // Update detailed statistics table with safe defaults
  document.getElementById("detailMeanRUL").textContent = `${(
    statistics.mean_rul_minutes || 0
  ).toFixed(2)} min`;
  document.getElementById("detailMedianRUL").textContent = `${(
    statistics.median_rul_minutes || 0
  ).toFixed(2)} min`;
  document.getElementById("minRUL").textContent = `${(
    statistics.min_rul_minutes || 0
  ).toFixed(2)} min`;
  document.getElementById("maxRUL").textContent = `${(
    statistics.max_rul_minutes || 0
  ).toFixed(2)} min`;
  document.getElementById("rangeRUL").textContent = `${(
    statistics.range_rul_minutes || 0
  ).toFixed(2)} min`;
  document.getElementById("stdRUL").textContent = `${(
    statistics.std_rul_minutes || 0
  ).toFixed(2)} min`;
  document.getElementById("varianceRUL").textContent = `${(
    statistics.variance_rul_minutes || 0
  ).toFixed(2)}`;
  document.getElementById("q1RUL").textContent = `${(
    statistics.q1_rul_minutes || 0
  ).toFixed(2)} min`;
  document.getElementById("q3RUL").textContent = `${(
    statistics.q3_rul_minutes || 0
  ).toFixed(2)} min`;
  document.getElementById("iqrRUL").textContent = `${(
    statistics.iqr_rul_minutes || 0
  ).toFixed(2)} min`;
  document.getElementById("healthyCount").textContent =
    statistics.healthy_count || 0;

  // Create charts
  createRULProgressionChart(predictions);
  createHealthDistributionChart(statistics);
  createRULHistogramChart(predictions.rul_minutes || []);
  createUrgencyChart(statistics);

  // Show alerts based on conditions
  const criticalCount = statistics.critical_count || 0;
  const severeCount = statistics.severe_count || 0;

  if (criticalCount > 0) {
    showAlert(
      "warning",
      `🚨 CRITICAL: ${criticalCount} sequences need immediate maintenance!`
    );
  } else if (severeCount > 0) {
    showAlert(
      "warning",
      `⚠️ WARNING: ${severeCount} sequences in severe condition. Plan maintenance soon.`
    );
  }
}

// ==================== CREATE CHARTS ====================

function createRULProgressionChart(predictions) {
  const ctx = document.getElementById("rulProgressionChart").getContext("2d");

  if (rulProgressionChart) {
    rulProgressionChart.destroy();
  }

  const data = predictions.rul_minutes || [];
  const indices = predictions.sequence_indices || [];

  if (data.length === 0) {
    console.warn("No data available for RUL progression chart");
    return;
  }

  // Set canvas width dynamically based on data points
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
    options: {
      responsive: true,
      maintainAspectRatio: true,
      aspectRatio: 3,
      plugins: {
        legend: {
          display: true,
          position: "top",
        },
        tooltip: {
          callbacks: {
            label: function (context) {
              const rul = context.parsed.y.toFixed(2);
              const hours = (context.parsed.y / 60).toFixed(2);
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
            font: {
              size: 14,
              weight: "bold",
            },
          },
          ticks: {
            callback: function (value) {
              return value.toFixed(0);
            },
          },
        },
        x: {
          title: {
            display: true,
            text: "Sequence Index",
            font: {
              size: 14,
              weight: "bold",
            },
          },
          ticks: {
            maxRotation: 0,
            autoSkip: true,
            maxTicksLimit: 20,
          },
        },
      },
    },
  });
}

function createHealthDistributionChart(statistics) {
  const ctx = document
    .getElementById("healthDistributionChart")
    .getContext("2d");

  if (healthDistributionChart) {
    healthDistributionChart.destroy();
  }

  const healthyCount = statistics.healthy_count || 0;
  const warningCount = statistics.warning_count || 0;
  const severeCount = statistics.severe_count || 0;
  const criticalCount = statistics.critical_count || 0;

  healthDistributionChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Healthy", "Warning", "Severe", "Critical"],
      datasets: [
        {
          data: [healthyCount, warningCount, severeCount, criticalCount],
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
            font: {
              size: 12,
            },
            generateLabels: function (chart) {
              const data = chart.data;
              if (data.labels.length && data.datasets.length) {
                return data.labels.map((label, i) => {
                  const value = data.datasets[0].data[i];
                  const total = data.datasets[0].data.reduce(
                    (a, b) => a + b,
                    0
                  );
                  const percentage =
                    total > 0 ? ((value / total) * 100).toFixed(1) : 0;
                  return {
                    text: `${label}: ${value} (${percentage}%)`,
                    fillStyle: data.datasets[0].backgroundColor[i],
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
            label: function (context) {
              const total = statistics.total_sequences || 1;
              const value = context.parsed;
              const percentage = ((value / total) * 100).toFixed(1);
              return `${context.label}: ${value} (${percentage}%)`;
            },
          },
        },
      },
    },
  });
}

function createRULHistogramChart(rulData) {
  const ctx = document.getElementById("rulHistogramChart").getContext("2d");

  if (rulHistogramChart) {
    rulHistogramChart.destroy();
  }

  if (!rulData || rulData.length === 0) {
    console.warn("No data available for RUL histogram");
    return;
  }

  // Create histogram bins
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
        legend: {
          display: false,
        },
        tooltip: {
          callbacks: {
            label: function (context) {
              const total = rulData.length;
              const percentage =
                total > 0 ? ((context.parsed.y / total) * 100).toFixed(1) : 0;
              return `Count: ${context.parsed.y} (${percentage}%)`;
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
            font: {
              size: 14,
              weight: "bold",
            },
          },
          ticks: {
            precision: 0,
          },
        },
        x: {
          title: {
            display: true,
            text: "RUL Range (minutes)",
            font: {
              size: 14,
              weight: "bold",
            },
          },
          ticks: {
            maxRotation: 45,
            minRotation: 45,
          },
        },
      },
    },
  });
}

function createUrgencyChart(statistics) {
  const ctx = document.getElementById("urgencyChart").getContext("2d");

  if (urgencyChart) {
    urgencyChart.destroy();
  }

  const healthyCount = statistics.healthy_count || 0;
  const warningCount = statistics.warning_count || 0;
  const severeCount = statistics.severe_count || 0;
  const criticalCount = statistics.critical_count || 0;
  const total = statistics.total_sequences || 1;

  urgencyChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Low", "Medium", "High", "Critical"],
      datasets: [
        {
          label: "Count",
          data: [healthyCount, warningCount, severeCount, criticalCount],
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
        legend: {
          display: false,
        },
        tooltip: {
          callbacks: {
            label: function (context) {
              const value = context.parsed.y;
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
            font: {
              size: 14,
              weight: "bold",
            },
          },
          ticks: {
            precision: 0,
          },
        },
        x: {
          title: {
            display: true,
            text: "Maintenance Urgency",
            font: {
              size: 14,
              weight: "bold",
            },
          },
        },
      },
    },
  });
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
