/**
 * Cupid AI — Archetype Explorer & Diagnostic Quiz Logic
 */

const ARCHETYPE_CENTROIDS = [
    { id: 0, name: "The Grounded Stabilizers", desc: "Reliable, structured, and empathetic. High Agreeableness and Conscientiousness.", agree: 0.68, con: 0.45, extra: 0.40 },
    { id: 1, name: "The Dynamic Visionaries", desc: "Ambitious, outgoing, and career-driven. High Extraversion and Conscientiousness.", agree: 0.60, con: 0.65, extra: 0.65 },
    { id: 2, name: "The Spontaneous Socializers", desc: "High spontaneity, present-focused, lively emotional style, and flexible planning.", agree: 0.29, con: 0.56, extra: 0.35 },
    { id: 3, name: "The Independent Thinkers", desc: "Introspective, analytical, and calm. High cognitive depth and quiet focus.", agree: 0.45, con: 0.31, extra: 0.61 }
];

function diagnoseArchetype() {
    const agree = parseFloat(document.getElementById('quiz_agree').value) / 10.0;
    const con = parseFloat(document.getElementById('quiz_con').value) / 10.0;
    const extra = parseFloat(document.getElementById('quiz_extra').value) / 10.0;

    let minDistance = Infinity;
    let closestArch = null;

    ARCHETYPE_CENTROIDS.forEach(arch => {
        const dist = Math.sqrt(
            Math.pow(arch.agree - agree, 2) +
            Math.pow(arch.con - con, 2) +
            Math.pow(arch.extra - extra, 2)
        );
        if (dist < minDistance) {
            minDistance = dist;
            closestArch = arch;
        }
    });

    const resultBox = document.getElementById('quizResult');
    const nameEl = document.getElementById('quizArchName');
    const descEl = document.getElementById('quizArchDesc');

    if (closestArch && resultBox) {
        nameEl.innerText = `${closestArch.id}. ${closestArch.name}`;
        descEl.innerText = `${closestArch.desc} (Proximity Score: ${(1 / (1 + minDistance)).toFixed(2)})`;
        resultBox.style.display = 'block';
    }
}
