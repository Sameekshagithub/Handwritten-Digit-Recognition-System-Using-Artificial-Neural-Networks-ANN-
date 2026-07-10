document.addEventListener("DOMContentLoaded", () => {
  const uploadZone = document.getElementById("uploadZone");
  const fileInput = document.getElementById("fileInput");
  const previewWrap = document.getElementById("previewWrap");
  const previewImg = document.getElementById("previewImg");
  const predictBtn = document.getElementById("predictBtn");
  const resetBtn = document.getElementById("resetBtn");
  const loading = document.getElementById("loading");
  const resultCard = document.getElementById("resultCard");
  const errorBox = document.getElementById("errorBox");
  const digitDisplay = document.getElementById("digitDisplay");
  const confidenceLabel = document.getElementById("confidenceLabel");
  const confidenceFg = document.getElementById("confidenceFg");
  const probBars = document.getElementById("probBars");

  let selectedFile = null;
  const CIRCUMFERENCE = 2 * Math.PI * 60; // r=60

  function showError(message) {
    errorBox.textContent = message;
    errorBox.style.display = "block";
  }

  function hideError() {
    errorBox.style.display = "none";
  }

  function handleFile(file) {
    if (!file) return;
    selectedFile = file;
    hideError();

    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      previewWrap.style.display = "block";
      predictBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  uploadZone.addEventListener("click", () => fileInput.click());

  uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("dragover");
  });

  uploadZone.addEventListener("dragleave", () => {
    uploadZone.classList.remove("dragover");
  });

  uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("dragover");
    if (e.dataTransfer.files.length) {
      fileInput.files = e.dataTransfer.files;
      handleFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) {
      handleFile(fileInput.files[0]);
    }
  });

  predictBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    hideError();
    resultCard.style.display = "none";
    loading.style.display = "block";
    predictBtn.disabled = true;

    const formData = new FormData();
    formData.append("digitImage", selectedFile);

    try {
      const response = await fetch("/predict", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();

      loading.style.display = "none";
      predictBtn.disabled = false;

      if (!data.success) {
        showError(data.error || "Prediction failed. Please try again.");
        return;
      }

      renderResult(data);
    } catch (err) {
      loading.style.display = "none";
      predictBtn.disabled = false;
      showError("Could not reach the server. Please try again.");
    }
  });

  function renderResult(data) {
    digitDisplay.textContent = data.digit;
    confidenceLabel.textContent = `${data.confidence}%`;

    const offset = CIRCUMFERENCE - (data.confidence / 100) * CIRCUMFERENCE;
    confidenceFg.style.strokeDasharray = CIRCUMFERENCE;
    confidenceFg.style.strokeDashoffset = CIRCUMFERENCE;
    requestAnimationFrame(() => {
      confidenceFg.style.strokeDashoffset = offset;
    });

    probBars.innerHTML = "";
    data.probabilities.forEach((p, idx) => {
      const row = document.createElement("div");
      row.className = "prob-row";
      row.innerHTML = `
        <span class="label">${idx}</span>
        <span class="track"><span class="fill" style="width:${p}%"></span></span>
        <span class="value">${p}%</span>
      `;
      probBars.appendChild(row);
    });

    resultCard.style.display = "block";
    resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  resetBtn.addEventListener("click", () => {
    selectedFile = null;
    fileInput.value = "";
    previewWrap.style.display = "none";
    resultCard.style.display = "none";
    predictBtn.disabled = true;
    hideError();
  });
});
