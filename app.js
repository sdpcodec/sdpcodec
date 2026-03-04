(function () {
  const loading = document.getElementById("loading");
  const noData = document.getElementById("no-data");
  const app = document.getElementById("app");

  async function loadSamples() {
    if (location.protocol === "file:") {
      console.error("Cannot load samples.json when opening as file://. Use the demo server.");
      return null;
    }
    try {
      const url = new URL("samples.json", location.href);
      url.searchParams.set("_", String(Date.now()));
      const res = await fetch(url);
      if (!res.ok) return null;
      return await res.json();
    } catch (e) {
      console.error("Failed to load samples.json:", e);
      return null;
    }
  }

  function escapeHtml(s) {
    const div = document.createElement("div");
    div.textContent = s;
    return div.innerHTML;
  }

  function renderSameSample(sample, sr) {
    const card = document.createElement("div");
    card.className = "sample-card";
    card.innerHTML = `
      <div class="sample-header">
        <span class="sample-id">${escapeHtml(sample.base_id)}</span>
      </div>
      <div class="gt-section">
        <div class="label">Ground Truth</div>
        <audio controls preload="metadata" src="${escapeHtml(sample.gt)}"></audio>
      </div>
    `;
    const modelsDiv = document.createElement("div");
    modelsDiv.className = "models-section";
    modelsDiv.innerHTML = "<div class=\"label\">Reconstruction (Same)</div>";
    if (sample.models) {
      for (const [modelName, paths] of Object.entries(sample.models)) {
        if (!paths.same) continue;
        const row = document.createElement("div");
        row.className = "model-row";
        row.innerHTML = `
          <span class="model-name">${escapeHtml(modelName)}</span>
          <div class="audio-cell">
            <audio controls preload="metadata" src="${escapeHtml(paths.same)}"></audio>
          </div>
        `;
        modelsDiv.appendChild(row);
      }
    }
    card.appendChild(modelsDiv);
    return card;
  }

  function renderVcSample(sample, sr) {
    const card = document.createElement("div");
    card.className = "sample-card";
    card.innerHTML = `
      <div class="sample-header">
        <span class="sample-id">${escapeHtml(sample.base_id)}</span>
      </div>
      <div class="gt-section">
        <div class="label">Ground Truth</div>
        <audio controls preload="metadata" src="${escapeHtml(sample.gt)}"></audio>
      </div>
      ${sample.ref ? `
      <div class="gt-section">
        <div class="label">Reference (target speaker)</div>
        <audio controls preload="metadata" src="${escapeHtml(sample.ref)}"></audio>
      </div>
      ` : ""}
    `;
    const modelsDiv = document.createElement("div");
    modelsDiv.className = "models-section";
    modelsDiv.innerHTML = "<div class=\"label\">Voice Conversion (VC)</div>";
    if (sample.models) {
      for (const [modelName, paths] of Object.entries(sample.models)) {
        if (!paths.vc) continue;
        const row = document.createElement("div");
        row.className = "model-row";
        row.innerHTML = `
          <span class="model-name">${escapeHtml(modelName)}</span>
          <div class="audio-cell">
            <audio controls preload="metadata" src="${escapeHtml(paths.vc)}"></audio>
          </div>
        `;
        modelsDiv.appendChild(row);
      }
    }
    card.appendChild(modelsDiv);
    return card;
  }

  function hasData(data) {
    if (!data || typeof data !== "object") return false;
    const same24 = data.same && data.same["24k"] && (data.same["24k"].samples || []);
    const same16 = data.same && data.same["16k"] && (data.same["16k"].samples || []);
    const vc24 = data.vc && data.vc["24k"] && (data.vc["24k"].samples || []);
    const vc16 = data.vc && data.vc["16k"] && (data.vc["16k"].samples || []);
    return same24.length > 0 || same16.length > 0 || vc24.length > 0 || vc16.length > 0;
  }

  function render(mode, sr) {
    const container = app.querySelector(".sample-list");
    container.innerHTML = "";
    const block = window.demoData[mode] && window.demoData[mode][sr];
    if (!block || !block.samples || block.samples.length === 0) {
      const empty = document.createElement("p");
      empty.className = "muted";
      empty.textContent = "No samples for this combination.";
      container.appendChild(empty);
      return;
    }
    const renderFn = mode === "same" ? renderSameSample : renderVcSample;
    block.samples.forEach((s) => container.appendChild(renderFn(s, sr)));
  }

  function switchMode(mode) {
    document.querySelectorAll(".mode-tab").forEach((t) => {
      t.classList.toggle("active", t.dataset.mode === mode);
    });
    window.demoData.currentMode = mode;
    render(mode, window.demoData.currentSr);
  }

  function switchSr(sr) {
    document.querySelectorAll(".sr-tab").forEach((t) => {
      t.classList.toggle("active", t.dataset.sr === sr);
    });
    window.demoData.currentSr = sr;
    render(window.demoData.currentMode, sr);
  }

  function showNoData(reason) {
    loading.classList.add("hidden");
    if (reason === "file") {
      noData.innerHTML = "<p><strong>Opened as file (file://).</strong> Browsers block loading <code>samples.json</code> from file.</p>" +
        "<p>Run from the project root: <code>bash demo/serve.sh</code><br>Then open: <a href=\"http://localhost:50004/\">http://localhost:50004/</a></p>";
    }
    noData.classList.remove("hidden");
  }

  async function init() {
    if (location.protocol === "file:") {
      showNoData("file");
      return;
    }
    const data = await loadSamples();
    if (!hasData(data)) {
      loading.classList.add("hidden");
      noData.classList.remove("hidden");
      return;
    }

    loading.classList.add("hidden");
    app.classList.remove("hidden");
    window.demoData = data;
    window.demoData.currentMode = "same";
    window.demoData.currentSr = (data.sample_rate_groups && data.sample_rate_groups[0]) || "24k";

    document.querySelectorAll(".mode-tab").forEach((t) => {
      t.addEventListener("click", () => switchMode(t.dataset.mode));
    });

    const srs = data.sample_rate_groups || ["24k", "16k"];
    document.querySelectorAll(".sr-tab").forEach((t) => {
      const sr = t.dataset.sr;
      t.classList.toggle("hidden", !srs.includes(sr));
      t.addEventListener("click", () => switchSr(sr));
    });

    switchSr(window.demoData.currentSr);
  }

  init();
})();
