const API_BASE_URLS = [
  "https://obesity-risk-prediction-system.onrender.com",
  "http://127.0.0.1:5000",
  "http://localhost:5000",
];
let ACTIVE_API_BASE_URL = null;
let deferredInstallPrompt = null;

function setupInstallPrompt() {
  const installSection = document.getElementById("installAppSection");
  const installButton = document.getElementById("installAppBtn");

  if (!installSection || !installButton) return;

  window.addEventListener("beforeinstallprompt", (event) => {
    event.preventDefault();
    deferredInstallPrompt = event;
    installSection.hidden = false;
  });

  installButton.addEventListener("click", async () => {
    if (!deferredInstallPrompt) return;

    deferredInstallPrompt.prompt();
    try {
      await deferredInstallPrompt.userChoice;
    } finally {
      deferredInstallPrompt = null;
      installSection.hidden = true;
    }
  });

  window.addEventListener("appinstalled", () => {
    deferredInstallPrompt = null;
    installSection.hidden = true;
  });
}

function recommendationByRisk(classIndex) {
  const baseAdvice = [
    "Maintain a balanced diet with regular meal timing.",
    "Track body weight weekly and monitor lifestyle trends.",
    "Prioritize at least 30 minutes of physical activity most days.",
    "Increase daily hydration and vegetable intake.",
  ];

  const highRiskAdvice = [
    "Reduce high-calorie and processed foods significantly.",
    "Use a structured exercise plan with professional guidance.",
    "Consult a healthcare professional or nutritionist for supervision.",
    "Limit sedentary screen time and improve sleep routine.",
  ];

  if (classIndex >= 4) {
    return [...baseAdvice, ...highRiskAdvice];
  }

  if (classIndex >= 2) {
    return [
      ...baseAdvice,
      "Increase moderate physical activity and reduce sugary snacks.",
      "Practice portion control and avoid frequent between-meal eating.",
    ];
  }

  return baseAdvice;
}

function parseFormValues(formElement) {
  const data = new FormData(formElement);
  const payload = {};

  for (const [key, value] of data.entries()) {
    payload[key] = Number(value);
  }

  // Accept both meters (e.g. 1.72) and centimeters (e.g. 172).
  if (Number.isFinite(payload.Height) && payload.Height > 3) {
    payload.Height = payload.Height / 100;
  }

  return payload;
}

async function requestPrediction(payload) {
  let lastError = null;

  const orderedBaseUrls = ACTIVE_API_BASE_URL
    ? [ACTIVE_API_BASE_URL, ...API_BASE_URLS.filter((url) => url !== ACTIVE_API_BASE_URL)]
    : API_BASE_URLS;

  for (const baseUrl of orderedBaseUrls) {
    try {
      const response = await fetch(`${baseUrl}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || "Prediction request failed.");
      }

      ACTIVE_API_BASE_URL = baseUrl;
      return result;
    } catch (error) {
      lastError = error;
    }
  }

  throw lastError || new Error("Could not connect to backend API.");
}

async function findHealthyApiBaseUrl() {
  const probeUrls = ACTIVE_API_BASE_URL
    ? [ACTIVE_API_BASE_URL, ...API_BASE_URLS.filter((url) => url !== ACTIVE_API_BASE_URL)]
    : API_BASE_URLS;

  for (const baseUrl of probeUrls) {
    try {
      const response = await fetch(`${baseUrl}/`, { method: "GET" });
      if (!response.ok) continue;

      ACTIVE_API_BASE_URL = baseUrl;
      return baseUrl;
    } catch {
      // Try next API URL.
    }
  }

  return null;
}

function setApiStatus(statusElement, state, text) {
  if (!statusElement) return;

  statusElement.classList.remove("status-checking", "status-online", "status-offline");
  statusElement.classList.add(state);
  statusElement.textContent = text;
}

async function handleLoginPage() {
  // New login system is now handled in login.html
  // This function is kept for backwards compatibility
  setupInstallPrompt();
}

async function handleFormPage() {
  const healthForm = document.getElementById("healthForm");
  if (!healthForm) return;

  const apiStatus = document.getElementById("apiStatus");
  const submitButton = healthForm.querySelector("button[type='submit']");

  if (submitButton) {
    submitButton.disabled = true;
    submitButton.textContent = "Checking backend...";
  }

  setApiStatus(apiStatus, "status-checking", "Checking backend connection...");

  const healthyUrl = await findHealthyApiBaseUrl();
  if (healthyUrl) {
    setApiStatus(apiStatus, "status-online", `Backend online (${healthyUrl})`);
    if (submitButton) {
      submitButton.disabled = false;
      submitButton.textContent = "Predict Obesity Risk";
    }
  } else {
    setApiStatus(apiStatus, "status-offline", "Backend offline. Start backend/app.py and refresh.");
    if (submitButton) {
      submitButton.disabled = true;
      submitButton.textContent = "Backend Offline";
    }
  }

  healthForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    if (!ACTIVE_API_BASE_URL) {
      const retryUrl = await findHealthyApiBaseUrl();
      if (!retryUrl) {
        setApiStatus(apiStatus, "status-offline", "Backend offline. Start backend/app.py and refresh.");
        alert("Backend is offline. Start backend and refresh this page.");
        return;
      }
      setApiStatus(apiStatus, "status-online", `Backend online (${retryUrl})`);
    }

    submitButton.disabled = true;
    submitButton.textContent = "Predicting...";

    try {
      const payload = parseFormValues(healthForm);
      
      // Allow direct access without login - user can submit anonymously or with login
      const userId = localStorage.getItem('user_id');
      const userName = localStorage.getItem('user_name');
      
      // If user is logged in, include their info. Otherwise submit anonymously.
      if (userId && userName) {
        payload.user_id = parseInt(userId);
        payload.user_name = userName;
      } else {
        // Anonymous submission when accessed directly without login
        payload.user_id = null;
        payload.user_name = "Anonymous User";
      }
      
      localStorage.setItem("lastQuestionnaire", JSON.stringify(payload));

      const result = await requestPrediction(payload);

      localStorage.setItem("predictionResult", JSON.stringify(result));
      window.location.href = "result.html";
    } catch (error) {
      alert(`Unable to fetch prediction: ${error.message}. Ensure backend is running at http://127.0.0.1:5000.`);
    } finally {
      submitButton.disabled = false;
      submitButton.textContent = "Predict Obesity Risk";
    }
  });
}

function riskBadgeMeta(classIndex) {
  if (classIndex >= 4) {
    return { label: "High Risk", className: "risk-high" };
  }

  if (classIndex >= 2) {
    return { label: "Moderate Risk", className: "risk-mid" };
  }

  return { label: "Lower Risk", className: "risk-low" };
}

function renderPredictionResult(result) {
  const riskLevelElement = document.getElementById("riskLevel");
  const confidenceElement = document.getElementById("confidence");
  const recommendationList = document.getElementById("recommendationList");
  const riskBadge = document.getElementById("riskBadge");
  const confidenceFill = document.getElementById("confidenceFill");

  if (!riskLevelElement || !confidenceElement || !recommendationList) return;

  riskLevelElement.textContent = result.predicted_risk_level;
  const percentage = Number(result.confidence || 0) * 100;
  confidenceElement.textContent = `Confidence: ${percentage.toFixed(2)}%`;

  if (confidenceFill) {
    confidenceFill.style.width = `${Math.min(Math.max(percentage, 0), 100).toFixed(0)}%`;
  }

  if (riskBadge) {
    const badge = riskBadgeMeta(Number(result.predicted_class_index));
    riskBadge.textContent = badge.label;
    riskBadge.classList.remove("risk-low", "risk-mid", "risk-high");
    riskBadge.classList.add(badge.className);
  }

  const recommendations = recommendationByRisk(Number(result.predicted_class_index));
  recommendationList.innerHTML = recommendations.map((item) => `<li>${item}</li>`).join("");
}

async function handleResultPage() {
  const result = JSON.parse(localStorage.getItem("predictionResult") || "null");
  const lastQuestionnaire = JSON.parse(localStorage.getItem("lastQuestionnaire") || "null");
  const userName = localStorage.getItem("user_name") || "Patient";
  const userId = localStorage.getItem("user_id");

  const userGreeting = document.getElementById("userGreeting");
  const riskLevelElement = document.getElementById("riskLevel");
  const confidenceElement = document.getElementById("confidence");
  const recommendationList = document.getElementById("recommendationList");

  if (!riskLevelElement || !confidenceElement || !recommendationList || !userGreeting) return;

  if (userName) {
    userGreeting.textContent = `Patient: ${userName}`;
  } else {
    userGreeting.textContent = "Prediction generated from your submitted questionnaire.";
  }

  if (result) {
    renderPredictionResult(result);
    return;
  }

  if (lastQuestionnaire) {
    try {
      riskLevelElement.textContent = "Generating prediction...";
      confidenceElement.textContent = "Please wait while we contact the backend.";
      const regeneratedResult = await requestPrediction(lastQuestionnaire);
      localStorage.setItem("predictionResult", JSON.stringify(regeneratedResult));
      renderPredictionResult(regeneratedResult);
      return;
    } catch {
      // Fallback to guidance below.
    }
  }

  if (!result) {
    riskLevelElement.textContent = "No prediction available";
    confidenceElement.textContent = "Please complete the questionnaire first.";
    recommendationList.innerHTML = "<li>Submit a new questionnaire to receive recommendations.</li>";
    return;
  }
}

(function init() {
  const page = document.body.dataset.page;

  if (page === "login") {
    handleLoginPage();
  }

  if (page === "form") {
    handleFormPage();
  }

  if (page === "result") {
    handleResultPage();
  }
})();
