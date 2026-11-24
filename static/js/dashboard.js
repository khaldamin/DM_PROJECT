/**
 * Cupid AI — Model Performance Dashboard Visualizer
 * Renders Canvas-based charts and updates metrics dynamically.
 */

document.addEventListener('DOMContentLoaded', () => {
    initModelComparisonChart();
    initArchetypePieChart();
    fetchModelMetadata();
});

/**
 * Fetch dynamic model metadata from /api/model-info if available.
 */
function fetchModelMetadata() {
    fetch('/api/model-info')
        .then(res => res.json())
        .then(data => {
            if (data && data.metrics) {
                if (data.metrics['ROC-AUC']) {
                    document.getElementById('val_roc_auc').innerText = Number(data.metrics['ROC-AUC']).toFixed(4);
                }
                if (data.metrics['Accuracy']) {
                    document.getElementById('val_accuracy').innerText = (Number(data.metrics['Accuracy']) * 100).toFixed(2) + '%';
                }
                if (data.metrics['F1-Score']) {
                    document.getElementById('val_f1').innerText = (Number(data.metrics['F1-Score']) * 100).toFixed(2) + '%';
                }
            }
        })
        .catch(err => {
            console.log('Using static benchmark metrics:', err);
        });
}

/**
 * Renders Horizontal Bar Chart comparing 5 models on ROC-AUC via HTML5 Canvas.
 */
function initModelComparisonChart() {
    const canvas = document.getElementById('modelComparisonChart');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();

    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const models = [
        { name: "Tuned XGBoost", score: 0.7969, highlight: true },
        { name: "XGBoost", score: 0.7960, highlight: false },
        { name: "Logistic Reg.", score: 0.7934, highlight: false },
        { name: "Random Forest", score: 0.7882, highlight: false },
        { name: "Decision Tree", score: 0.7253, highlight: false }
    ];

    const width = rect.width;
    const height = rect.height;
    const paddingLeft = 110;
    const paddingRight = 60;
    const barHeight = 28;
    const gap = (height - (models.length * barHeight)) / (models.length + 1);

    ctx.clearRect(0, 0, width, height);

    models.forEach((m, i) => {
        const y = gap + i * (barHeight + gap);
        const maxScore = 0.85;
        const barWidth = ((m.score - 0.65) / (maxScore - 0.65)) * (width - paddingLeft - paddingRight);

        // Label
        ctx.fillStyle = m.highlight ? "#ffffff" : "rgba(255, 255, 255, 0.7)";
        ctx.font = m.highlight ? "bold 13px 'Plus Jakarta Sans', sans-serif" : "12px 'Plus Jakarta Sans', sans-serif";
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        ctx.fillText(m.name, paddingLeft - 12, y + barHeight / 2);

        // Bar background track
        ctx.fillStyle = "rgba(255, 255, 255, 0.05)";
        ctx.beginPath();
        ctx.roundRect(paddingLeft, y, width - paddingLeft - paddingRight, barHeight, 6);
        ctx.fill();

        // Bar fill
        if (m.highlight) {
            const grad = ctx.createLinearGradient(paddingLeft, 0, paddingLeft + barWidth, 0);
            grad.addColorStop(0, '#ff758c');
            grad.addColorStop(1, '#ff7eb3');
            ctx.fillStyle = grad;
        } else {
            ctx.fillStyle = "rgba(255, 255, 255, 0.25)";
        }

        ctx.beginPath();
        ctx.roundRect(paddingLeft, y, Math.max(10, barWidth), barHeight, 6);
        ctx.fill();

        // Score text
        ctx.fillStyle = m.highlight ? "#ff7eb3" : "#ffffff";
        ctx.font = "bold 12px 'Outfit', sans-serif";
        ctx.textAlign = "left";
        ctx.fillText(m.score.toFixed(4), paddingLeft + barWidth + 8, y + barHeight / 2);
    });
}

/**
 * Renders Donut/Pie Chart of Archetype Population via HTML5 Canvas.
 */
function initArchetypePieChart() {
    const canvas = document.getElementById('archetypeDistributionChart');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();

    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const slices = [
        { label: "Grounded Stabilizers", value: 23.3, color: "#27ae60" },
        { label: "Dynamic Visionaries", value: 25.7, color: "#8e44ad" },
        { label: "Spontaneous Socializers", value: 25.9, color: "#d35400" },
        { label: "Independent Thinkers", value: 25.1, color: "#2980b9" }
    ];

    const centerX = rect.width * 0.35;
    const centerY = rect.height * 0.5;
    const radius = Math.min(centerX, centerY) * 0.75;
    const innerRadius = radius * 0.5;

    let startAngle = -Math.PI / 2;

    slices.forEach(slice => {
        const sliceAngle = (slice.value / 100) * (2 * Math.PI);
        const endAngle = startAngle + sliceAngle;

        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, startAngle, endAngle);
        ctx.arc(centerX, centerY, innerRadius, endAngle, startAngle, true);
        ctx.closePath();
        ctx.fillStyle = slice.color;
        ctx.fill();

        startAngle = endAngle;
    });

    // Render Legend
    const legendX = rect.width * 0.65;
    let legendY = centerY - (slices.length * 24) / 2;

    slices.forEach(slice => {
        ctx.fillStyle = slice.color;
        ctx.beginPath();
        ctx.roundRect(legendX, legendY, 12, 12, 3);
        ctx.fill();

        ctx.fillStyle = "#ffffff";
        ctx.font = "11px 'Plus Jakarta Sans', sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(`${slice.label} (${slice.value}%)`, legendX + 18, legendY + 6);

        legendY += 26;
    });
}
