
document.addEventListener("DOMContentLoaded", function () {
    const canvas = document.getElementById("archetypesRadarChart");

    if (!canvas || typeof Chart === "undefined") {
        console.error("Radar chart canvas or Chart.js not found.");
        return;
    }

    const labels = [
        "Openness",
        "Extraversion",
        "Agreeableness",
        "Conscientiousness",
        "Chronotype",
        "Spontaneity",
        "Emotional Expression",
        "Career Ambition"
    ];

    // Real centroid values from archetypes.py / cluster_archetypes.csv.
    // Original values are normalized between 0 and 1.
    // Multiply by 10 to display them on a 0–10 scale.
    const datasets = [
        {
            label: "Grounded Stabilizers",
            data: [5.0, 4.0, 6.8, 4.5, 4.4, 5.3, 3.2, 4.8],
            borderColor: "#38bdf8",
            backgroundColor: "rgba(56, 189, 248, 0.12)",
            pointBackgroundColor: "#38bdf8"
        },
        {
            label: "Dynamic Visionaries",
            data: [4.7, 6.5, 6.0, 6.5, 6.0, 4.0, 6.1, 4.9],
            borderColor: "#a78bfa",
            backgroundColor: "rgba(167, 139, 250, 0.12)",
            pointBackgroundColor: "#a78bfa"
        },
        {
            label: "Spontaneous Socializers",
            data: [5.2, 3.5, 2.9, 5.6, 5.4, 4.4, 4.4, 5.1],
            borderColor: "#fb923c",
            backgroundColor: "rgba(251, 146, 60, 0.12)",
            pointBackgroundColor: "#fb923c"
        },
        {
            label: "Independent Thinkers",
            data: [5.0, 6.1, 4.5, 3.1, 4.2, 6.3, 6.1, 5.1],
            borderColor: "#34d399",
            backgroundColor: "rgba(52, 211, 153, 0.12)",
            pointBackgroundColor: "#34d399"
        }
    ];

    new Chart(canvas, {
        type: "radar",

        data: {
            labels: labels,
            datasets: datasets.map(dataset => ({
                ...dataset,
                borderWidth: 2.5,
                pointRadius: 3,
                pointHoverRadius: 6,
                pointBorderWidth: 1,
                fill: true
            }))
        },

        options: {
            responsive: true,
            maintainAspectRatio: false,

            interaction: {
                mode: "nearest",
                intersect: false
            },

            plugins: {
                legend: {
                    position: "bottom",
                    labels: {
                        color: "#e2e8f0",
                        padding: 20,
                        usePointStyle: true,
                        pointStyle: "circle",
                        font: {
                            family: "Outfit",
                            size: 12
                        }
                    }
                },

                tooltip: {
                    backgroundColor: "rgba(15, 23, 42, 0.95)",
                    titleColor: "#ffffff",
                    bodyColor: "#cbd5e1",
                    borderColor: "rgba(255,255,255,0.15)",
                    borderWidth: 1,
                    padding: 12,

                    callbacks: {
                        label: function (context) {
                            return context.dataset.label + ": " +
                                context.parsed.r.toFixed(1) + " / 10";
                        }
                    }
                }
            },

            scales: {
                r: {
                    min: 0,
                    max: 10,

                    ticks: {
                        stepSize: 2,
                        color: "#94a3b8",
                        backdropColor: "transparent",
                        font: {
                            size: 10
                        }
                    },

                    pointLabels: {
                        color: "#e2e8f0",
                        font: {
                            family: "Outfit",
                            size: 11,
                            weight: "500"
                        }
                    },

                    grid: {
                        color: "rgba(148, 163, 184, 0.18)"
                    },

                    angleLines: {
                        color: "rgba(148, 163, 184, 0.2)"
                    }
                }
            }
        }
    });
});
