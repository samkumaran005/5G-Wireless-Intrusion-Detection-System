/**
 * 5G-WIDS FRONTEND CONTROLLER
 * Handles Tab Navigation, Drag & Drop, API Communication,
 * Chart.js Visualizations, and Flow Table Rendering.
 */

// State variables
let currentSelectedFile = null;
let currentAnalysisResults = null;
let attackChartInstance = null;
let threatBarChartInstance = null;
let allFlowRecords = [];

// Initialize on DOM load
document.addEventListener("DOMContentLoaded", () => {
    initDragAndDrop();
    loadFeaturesDirectory();
});

/* ==========================================================================
   NAVIGATION & TABS
   ========================================================================== */
function switchTab(tabId) {
    // Update nav buttons
    document.querySelectorAll(".nav-tab").forEach(btn => btn.classList.remove("active"));
    const activeBtn = document.getElementById(`tab-${tabId}-btn`);
    if (activeBtn) activeBtn.classList.add("active");

    // Update section visibility
    document.querySelectorAll(".tab-content").forEach(sec => sec.classList.remove("active"));
    const activeSection = document.getElementById(`section-${tabId}`);
    if (activeSection) activeSection.classList.add("active");

    window.scrollTo({ top: 0, behavior: "smooth" });
}

function switchViz(vizId) {
    document.querySelectorAll(".viz-tab-btn").forEach(btn => btn.classList.remove("active"));
    event.target.classList.add("active");

    document.querySelectorAll(".viz-panel").forEach(panel => panel.classList.remove("active"));
    const targetPanel = document.getElementById(`viz-panel-${vizId}`);
    if (targetPanel) targetPanel.classList.add("active");
}

/* ==========================================================================
   91 FEATURES DIRECTORY
   ========================================================================== */
async function loadFeaturesDirectory() {
    try {
        const response = await fetch("/api/features");
        const data = await response.json();
        const container = document.getElementById("feature-groups-container");
        if (!container) return;

        container.innerHTML = "";
        for (const [groupName, features] of Object.entries(data.groups)) {
            const card = document.createElement("div");
            card.className = "feature-group-card";
            card.innerHTML = `
                <div class="group-title">
                    <span>${groupName}</span>
                    <span class="group-count">${features.length} Features</span>
                </div>
                <div class="feature-tag-list">
                    ${features.map(f => `<span class="feat-tag" title="${f}">${f}</span>`).join("")}
                </div>
            `;
            container.appendChild(card);
        }
    } catch (err) {
        console.error("Failed to load feature groups:", err);
    }
}

function filterFeatures() {
    const query = document.getElementById("feature-filter-input").value.toLowerCase();
    const tags = document.querySelectorAll(".feat-tag");
    tags.forEach(tag => {
        const text = tag.textContent.toLowerCase();
        tag.style.display = text.includes(query) ? "inline-block" : "none";
    });
}

/* ==========================================================================
   FILE SELECTION & DRAG AND DROP
   ========================================================================== */
function initDragAndDrop() {
    const dropZone = document.getElementById("drop-zone");
    if (!dropZone) return;

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });
}

function handleFileSelect(event) {
    const files = event.target.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFile(file) {
    if (!file.name.toLowerCase().endsWith(".csv")) {
        alert("Please upload a valid .CSV network traffic file.");
        return;
    }

    currentSelectedFile = file;
    const fileChip = document.getElementById("file-info-chip");
    const nameEl = document.getElementById("selected-file-name");
    const sizeEl = document.getElementById("selected-file-size");

    nameEl.textContent = file.name;
    sizeEl.textContent = (file.size / 1024).toFixed(1) + " KB";
    fileChip.style.display = "flex";
}

function clearSelectedFile(e) {
    if (e) e.stopPropagation();
    currentSelectedFile = null;
    document.getElementById("file-input").value = "";
    document.getElementById("file-info-chip").style.display = "none";
}

/* ==========================================================================
   LIVE TRAFFIC ANALYSIS TRIGGER
   ========================================================================== */
async function loadSampleTraffic() {
    // Select sample dataset mode
    currentSelectedFile = "SAMPLE_5G_TRAFFIC";
    const fileChip = document.getElementById("file-info-chip");
    const nameEl = document.getElementById("selected-file-name");
    const sizeEl = document.getElementById("selected-file-size");

    nameEl.textContent = "sample_5g_traffic.csv (Ready)";
    sizeEl.textContent = "100 Live Flow Records &bull; 97 Columns";
    fileChip.style.display = "flex";

    // Trigger analysis
    runTrafficAnalysis();
}

async function runTrafficAnalysis() {
    if (!currentSelectedFile) {
        alert("Please select or drop a CSV network traffic file first, or click 'Load Sample 5G Traffic'.");
        return;
    }

    const btnAnalyze = document.getElementById("btn-analyze");
    const loadingBox = document.getElementById("analysis-loading");
    const loadingStage = document.getElementById("loading-stage");

    btnAnalyze.disabled = true;
    loadingBox.style.display = "flex";

    // Progressive stage animation
    const stages = [
        "Aligning 91 Network Features & Normalizing...",
        "Building k-NN Topological Flow Graphs (k=2)...",
        "Constructing Temporal Graph Sequences (W=5)...",
        "Executing HTSTCL-GNN Multi-Head Attention Inference..."
    ];
    let stageIdx = 0;
    const stageInterval = setInterval(() => {
        stageIdx = (stageIdx + 1) % stages.length;
        if (loadingStage) loadingStage.textContent = stages[stageIdx];
    }, 600);

    try {
        let response;
        if (currentSelectedFile === "SAMPLE_5G_TRAFFIC") {
            response = await fetch("/api/analyze", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ use_sample: true })
            });
        } else {
            const formData = new FormData();
            formData.append("file", currentSelectedFile);
            response = await fetch("/api/analyze", {
                method: "POST",
                body: formData
            });
        }

        const data = await response.json();
        clearInterval(stageInterval);

        if (data.error) {
            alert("Analysis Error: " + data.error);
            return;
        }

        currentAnalysisResults = data;
        renderAnalysisResults(data);

    } catch (err) {
        clearInterval(stageInterval);
        console.error("Traffic analysis request failed:", err);
        alert("Failed to analyze traffic. Please check console logs.");
    } finally {
        btnAnalyze.disabled = false;
        loadingBox.style.display = "none";
    }
}

/* ==========================================================================
   RENDER ANALYSIS RESULTS
   ========================================================================== */
function renderAnalysisResults(results) {
    const container = document.getElementById("results-container");
    container.style.display = "block";

    // 1. Update Threat Banner
    const banner = document.getElementById("results-banner");
    const bannerTitle = document.getElementById("banner-title");
    const bannerDesc = document.getElementById("banner-desc");
    const bannerTag = document.getElementById("threat-level-tag");
    const bannerIcon = document.getElementById("banner-status-icon");

    if (results.threat_level === "CRITICAL" || results.threat_level === "ELEVATED") {
        banner.className = "results-header-banner";
        bannerTitle.textContent = `Threat Assessment: ${results.threat_level}`;
        bannerDesc.textContent = `Detected ${results.attacks_detected} intrusions (${results.attack_percentage}% of flow records). Action recommended.`;
        bannerTag.textContent = results.threat_level;
        bannerIcon.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i>`;
    } else {
        banner.className = "results-header-banner banner-secure";
        bannerTitle.textContent = "Threat Assessment: SECURE";
        bannerDesc.textContent = "No intrusion anomalies detected. 5G User plane and control traffic verified normal.";
        bannerTag.textContent = "SECURE";
        bannerIcon.innerHTML = `<i class="fa-solid fa-shield-check"></i>`;
    }

    // 2. Update Top 4 KPI Cards
    document.getElementById("kpi-total-records").textContent = results.total_records.toLocaleString();
    document.getElementById("kpi-seq-count").textContent = `${results.sequences_analyzed} Temporal Sequences (${results.graphs_constructed} Graphs)`;

    document.getElementById("kpi-attacks-detected").textContent = results.attacks_detected.toLocaleString();
    document.getElementById("kpi-attack-pct").textContent = `${results.attack_percentage}% of total flows`;

    document.getElementById("kpi-normal-count").textContent = results.normal_traffic_count.toLocaleString();
    document.getElementById("kpi-normal-pct").textContent = `${results.normal_percentage}% of total flows`;

    document.getElementById("kpi-avg-confidence").textContent = `${results.avg_confidence}%`;

    // 3. Render Chart.js Visualizations
    renderAttackDonutChart(results.attack_distribution);
    renderThreatBarChart(results);

    // 4. Render Detailed Flow Inspection Table
    allFlowRecords = results.flow_records || [];
    renderFlowsTable(allFlowRecords);

    // Smooth scroll to results
    container.scrollIntoView({ behavior: "smooth", block: "start" });
}

/* ==========================================================================
   CHART.JS VISUALIZATIONS
   ========================================================================== */
function renderAttackDonutChart(distribution) {
    const ctx = document.getElementById("attackDistributionChart");
    if (!ctx) return;

    if (attackChartInstance) {
        attackChartInstance.destroy();
    }

    const labels = Object.keys(distribution);
    const dataVals = Object.values(distribution);

    // Color palette based on labels
    const colors = labels.map(label => {
        const lower = label.toLowerCase();
        if (lower.includes("benign") || lower.includes("normal")) return "#10b981"; // green
        if (lower.includes("synscan")) return "#ef4444"; // red
        if (lower.includes("ddos") || lower.includes("dos")) return "#f43f5e"; // bright red
        if (lower.includes("portscan") || lower.includes("probe")) return "#f59e0b"; // amber
        return "#a855f7"; // purple for anomalous
    });

    attackChartInstance = new Chart(ctx, {
        type: "doughnut",
        data: {
            labels: labels,
            datasets: [{
                data: dataVals,
                backgroundColor: colors,
                borderColor: "#0a0f1d",
                borderWidth: 3,
                hoverOffset: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: "right",
                    labels: {
                        color: "#94a3b8",
                        font: { family: "'Inter', sans-serif", size: 12, weight: "600" },
                        padding: 16,
                        boxWidth: 14
                    }
                },
                tooltip: {
                    backgroundColor: "rgba(15, 23, 42, 0.95)",
                    titleColor: "#00f2fe",
                    bodyColor: "#f8fafc",
                    borderColor: "rgba(0, 242, 254, 0.3)",
                    borderWidth: 1,
                    padding: 12,
                    callbacks: {
                        label: function(context) {
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const val = context.raw;
                            const pct = ((val / total) * 100).toFixed(1);
                            return ` ${context.label}: ${val} flows (${pct}%)`;
                        }
                    }
                }
            },
            cutout: "68%"
        }
    });
}

function renderThreatBarChart(results) {
    const ctx = document.getElementById("threatBreakdownBarChart");
    if (!ctx) return;

    if (threatBarChartInstance) {
        threatBarChartInstance.destroy();
    }

    const labels = ["Normal Flows", "Detected Attacks"];
    const values = [results.normal_traffic_count, results.attacks_detected];

    threatBarChartInstance = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: "Flow Count",
                data: values,
                backgroundColor: ["rgba(16, 185, 129, 0.75)", "rgba(239, 68, 68, 0.75)"],
                borderColor: ["#10b981", "#ef4444"],
                borderWidth: 1.5,
                borderRadius: 8,
                barThickness: 45
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: "rgba(15, 23, 42, 0.95)",
                    titleColor: "#fff",
                    bodyColor: "#f8fafc",
                    borderColor: "rgba(255, 255, 255, 0.1)",
                    borderWidth: 1,
                    padding: 10
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: "#94a3b8", font: { family: "'Inter', sans-serif", weight: "600" } }
                },
                y: {
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    ticks: { color: "#64748b", font: { family: "'JetBrains Mono', monospace" } }
                }
            }
        }
    });
}

/* ==========================================================================
   FLOWS INSPECTION TABLE
   ========================================================================== */
function renderFlowsTable(records) {
    const tbody = document.getElementById("flows-table-body");
    if (!tbody) return;

    tbody.innerHTML = "";

    if (records.length === 0) {
        tbody.innerHTML = `<tr><td colspan="9" style="text-align:center; padding: 2rem; color: #64748b;">No flow records match the selected filter.</td></tr>`;
        return;
    }

    records.forEach(flow => {
        const tr = document.createElement("tr");

        const predBadge = flow.is_attack
            ? `<span class="badge-pred badge-pred-malicious"><i class="fa-solid fa-triangle-exclamation"></i> MALICIOUS</span>`
            : `<span class="badge-pred badge-pred-benign"><i class="fa-solid fa-check"></i> BENIGN</span>`;

        let sevClass = "badge-sev-normal";
        if (flow.threat_severity === "HIGH") sevClass = "badge-sev-high";
        else if (flow.threat_severity === "MEDIUM") sevClass = "badge-sev-medium";

        let atkCategoryBadge = `<span style="font-weight: 600; color: #94a3b8;">${flow.attack_type}</span>`;
        if (flow.is_attack) {
            atkCategoryBadge = `<span style="font-weight: 700; color: #f43f5e; background: rgba(244, 63, 94, 0.12); padding: 3px 8px; border-radius: 4px; border: 1px solid rgba(244, 63, 94, 0.3);"><i class="fa-solid fa-bug"></i> ${flow.attack_type}</span>`;
        } else {
            atkCategoryBadge = `<span style="font-weight: 600; color: #10b981;"><i class="fa-solid fa-shield-halved"></i> Benign</span>`;
        }

        tr.innerHTML = `
            <td>#${flow.flow_id}</td>
            <td>${flow.dur}</td>
            <td>${flow.tot_pkts}</td>
            <td>${flow.tot_bytes.toLocaleString()}</td>
            <td>${flow.rate}</td>
            <td>${predBadge}</td>
            <td style="color: ${flow.is_attack ? '#ef4444' : '#10b981'}; font-weight: 700;">${flow.confidence}%</td>
            <td>${atkCategoryBadge}</td>
            <td class="${sevClass}">${flow.threat_severity}</td>
        `;
        tbody.appendChild(tr);
    });
}

function filterFlows(type) {
    document.querySelectorAll(".filter-btn").forEach(btn => btn.classList.remove("active"));
    event.target.classList.add("active");

    if (type === "ALL") {
        renderFlowsTable(allFlowRecords);
    } else if (type === "MALICIOUS") {
        renderFlowsTable(allFlowRecords.filter(r => r.is_attack));
    } else if (type === "BENIGN") {
        renderFlowsTable(allFlowRecords.filter(r => !r.is_attack));
    }
}
