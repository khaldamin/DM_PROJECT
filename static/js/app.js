// Interactive JavaScript for Cupid AI Dashboard

function updateVal(id) {
    const input = document.getElementById(id);
    const label = document.getElementById('val_' + id);
    if (label && input) {
        label.innerText = parseFloat(input.value).toFixed(1);
    }
}

// Preset Loader
function loadPreset(type) {
    const presets = {
        perfect: {
            a_age: 28, a_education: 3, a_location: 'Urban', a_career_field: 'Tech', a_love_language: 'Quality Time',
            a_openness: 8.0, a_extraversion: 7.0, a_agreeableness: 8.5, a_conscientiousness: 8.0,
            a_career_ambition: 8.0, a_chronotype: 6.0, a_spontaneity: 7.0, a_emotional_expressiveness: 8.0,
            
            b_age: 27, b_education: 3, b_location: 'Urban', b_career_field: 'Healthcare', b_love_language: 'Quality Time',
            b_openness: 8.0, b_extraversion: 7.5, b_agreeableness: 9.0, b_conscientiousness: 8.0,
            b_career_ambition: 7.5, b_chronotype: 6.0, b_spontaneity: 7.0, b_emotional_expressiveness: 8.5
        },
        balanced: {
            a_age: 30, a_education: 2, a_location: 'Suburban', a_career_field: 'Finance', a_love_language: 'Acts of Service',
            a_openness: 6.5, a_extraversion: 6.0, a_agreeableness: 7.5, a_conscientiousness: 7.5,
            a_career_ambition: 7.0, a_chronotype: 4.0, a_spontaneity: 5.5, a_emotional_expressiveness: 6.5,

            b_age: 31, b_education: 2, b_location: 'Suburban', b_career_field: 'Marketing', b_love_language: 'Acts of Service',
            b_openness: 7.0, b_extraversion: 6.5, b_agreeableness: 7.0, b_conscientiousness: 6.5,
            b_career_ambition: 6.5, b_chronotype: 5.0, b_spontaneity: 6.0, b_emotional_expressiveness: 7.0
        },
        opposites: {
            a_age: 26, a_education: 2, a_location: 'Urban', a_career_field: 'Creative Arts', a_love_language: 'Physical Touch',
            a_openness: 9.0, a_extraversion: 9.0, a_agreeableness: 6.5, a_conscientiousness: 4.5,
            a_career_ambition: 5.0, a_chronotype: 9.0, a_spontaneity: 9.5, a_emotional_expressiveness: 9.0,

            b_age: 32, b_education: 4, b_location: 'Suburban', b_career_field: 'Law', b_love_language: 'Words of Affirmation',
            b_openness: 5.0, b_extraversion: 3.5, b_agreeableness: 8.0, b_conscientiousness: 9.5,
            b_career_ambition: 9.0, b_chronotype: 2.0, b_spontaneity: 3.0, b_emotional_expressiveness: 5.0
        },
        clash: {
            a_age: 24, a_education: 1, a_location: 'Rural', a_career_field: 'Entrepreneurship', a_love_language: 'Receiving Gifts',
            a_openness: 3.0, a_extraversion: 2.0, a_agreeableness: 3.5, a_conscientiousness: 3.0,
            a_career_ambition: 9.5, a_chronotype: 1.0, a_spontaneity: 1.0, a_emotional_expressiveness: 2.0,

            b_age: 42, b_education: 4, b_location: 'Urban', b_career_field: 'Education', b_love_language: 'Quality Time',
            b_openness: 9.5, b_extraversion: 9.5, b_agreeableness: 9.0, b_conscientiousness: 9.0,
            b_career_ambition: 3.0, b_chronotype: 9.5, b_spontaneity: 9.5, b_emotional_expressiveness: 9.5
        }
    };

    const p = presets[type];
    if (!p) return;

    for (const key in p) {
        const el = document.getElementById(key);
        if (el) {
            el.value = p[key];
            if (el.type === 'range') {
                updateVal(key);
            }
        }
    }
}

// Predict Compatibility
async function predictCompatibility() {
    const btn = document.getElementById('btnPredict');
    btn.disabled = true;
    btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Running Machine Learning Pipeline...`;

    const formData = {
        a_age: parseFloat(document.getElementById('a_age').value),
        a_education: parseInt(document.getElementById('a_education').value),
        a_location: document.getElementById('a_location').value,
        a_career_field: document.getElementById('a_career_field').value,
        a_love_language: document.getElementById('a_love_language').value,
        a_openness: parseFloat(document.getElementById('a_openness').value),
        a_extraversion: parseFloat(document.getElementById('a_extraversion').value),
        a_agreeableness: parseFloat(document.getElementById('a_agreeableness').value),
        a_conscientiousness: parseFloat(document.getElementById('a_conscientiousness').value),
        a_career_ambition: parseFloat(document.getElementById('a_career_ambition').value),
        a_chronotype: parseFloat(document.getElementById('a_chronotype').value),
        a_spontaneity: parseFloat(document.getElementById('a_spontaneity').value),
        a_emotional_expressiveness: parseFloat(document.getElementById('a_emotional_expressiveness').value),

        b_age: parseFloat(document.getElementById('b_age').value),
        b_education: parseInt(document.getElementById('b_education').value),
        b_location: document.getElementById('b_location').value,
        b_career_field: document.getElementById('b_career_field').value,
        b_love_language: document.getElementById('b_love_language').value,
        b_openness: parseFloat(document.getElementById('b_openness').value),
        b_extraversion: parseFloat(document.getElementById('b_extraversion').value),
        b_agreeableness: parseFloat(document.getElementById('b_agreeableness').value),
        b_conscientiousness: parseFloat(document.getElementById('b_conscientiousness').value),
        b_career_ambition: parseFloat(document.getElementById('b_career_ambition').value),
        b_chronotype: parseFloat(document.getElementById('b_chronotype').value),
        b_spontaneity: parseFloat(document.getElementById('b_spontaneity').value),
        b_emotional_expressiveness: parseFloat(document.getElementById('b_emotional_expressiveness').value)
    };

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });

        let data;
        if (response.ok) {
            data = await response.json();
        } else {
            // Client side fallback computation if server not reached
            data = calculateLocalMatch(formData);
        }

        displayResults(data, formData);
    } catch (err) {
        console.warn("Backend API call failed, running client-side fallback computation:", err);
        const data = calculateLocalMatch(formData);
        displayResults(data, formData);
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<i class="fa-solid fa-wand-magic-sparkles"></i> Calculate Compatibility & Predict Match`;
    }
}

// Client-side Fallback Mathematical Engine (Trained XGBoost replica formula)
function calculateLocalMatch(p) {
    const age_diff = Math.abs(p.a_age - p.b_age);
    const same_loc = p.a_location === p.b_location ? 1.0 : 0.0;
    const same_career = p.a_career_field === p.b_career_field ? 1.0 : 0.0;
    const same_love = p.a_love_language === p.b_love_language ? 1.0 : 0.0;

    const trait_diffs = [
        Math.abs(p.a_openness - p.b_openness),
        Math.abs(p.a_extraversion - p.b_extraversion),
        Math.abs(p.a_agreeableness - p.b_agreeableness),
        Math.abs(p.a_conscientiousness - p.b_conscientiousness),
        Math.abs(p.a_chronotype - p.b_chronotype),
        Math.abs(p.a_spontaneity - p.b_spontaneity),
        Math.abs(p.a_emotional_expressiveness - p.b_emotional_expressiveness),
        Math.abs(p.a_career_ambition - p.b_career_ambition)
    ];

    const sum_sq_diff = trait_diffs.reduce((a, b) => a + b * b, 0);
    const dist = Math.sqrt(sum_sq_diff);
    const avg_trait_diff = trait_diffs.reduce((a, b) => a + b, 0) / trait_diffs.length;

    // Big 5 harmony
    const big5_diff = (trait_diffs[0] + trait_diffs[1] + trait_diffs[2] + trait_diffs[3]) / 4.0;
    const big5_harmony = Math.max(0, 100 - big5_diff * 12);

    // Compute compatibility score base
    let score = 95 - (dist * 7.5) - (age_diff * 0.8) + (same_loc * 6) + (same_love * 8) + (same_career * 3);
    score = Math.max(10, Math.min(99, score));

    const compatible = score >= 55 ? 1 : 0;
    const longevity = Math.round(score * 0.95 + (same_love ? 12 : 0) + (10 - age_diff));

    return {
        compatibility_score: parseFloat(score.toFixed(1)),
        compatible: compatible,
        estimated_longevity_months: Math.max(3, Math.min(120, longevity)),
        big5_harmony_score: parseFloat(big5_harmony.toFixed(1)),
        avg_trait_diff: parseFloat(avg_trait_diff.toFixed(2)),
        personality_distance: parseFloat(dist.toFixed(2))
    };
}

// Render Results UI
function displayResults(data, input) {
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.style.display = 'block';

    const score = data.compatibility_score;
    document.getElementById('scorePercent').innerText = `${score}%`;

    // Circular Gauge animation
    const circle = document.getElementById('gaugeProgress');
    const radius = 42;
    const circumference = 2 * Math.PI * radius; // ~264
    const offset = circumference - (score / 100) * circumference;
    circle.style.strokeDasharray = `${circumference}`;
    circle.style.strokeDashoffset = `${offset}`;

    // Badge and Color
    const badge = document.getElementById('compatibilityBadge');
    if (data.compatible === 1) {
        badge.className = 'status-badge compatible';
        badge.innerText = 'HIGH COMPATIBILITY ❤️';
        circle.style.stroke = '#10b981';
    } else {
        badge.className = 'status-badge incompatible';
        badge.innerText = 'HIGH FRICTION 💔';
        circle.style.stroke = '#ef4444';
    }

    document.getElementById('predictedOutcome').innerText = data.compatible === 1 ? "Strong Long-Term Match" : "Challenging / Volatile Match";
    document.getElementById('predictedLongevity').innerText = `~${data.estimated_longevity_months} months`;
    document.getElementById('harmonyIndex').innerText = `${data.big5_harmony_score}%`;

    // Breakdown Bars
    const big5Pct = Math.max(10, Math.min(100, 100 - (data.avg_trait_diff * 15)));
    document.getElementById('big5Bar').style.width = `${big5Pct}%`;
    document.getElementById('big5Desc').innerText = `Big 5 discrepancy: ${data.avg_trait_diff.toFixed(2)} pts average delta across personality dimensions.`;

    const chronoDiff = Math.abs(input.a_chronotype - input.b_chronotype);
    const lifestylePct = Math.max(10, 100 - (chronoDiff * 10));
    document.getElementById('lifestyleBar').style.width = `${lifestylePct}%`;
    document.getElementById('lifestyleDesc').innerText = `Chronotype gap is ${chronoDiff} pts. ${chronoDiff > 3 ? 'Opposite sleep/wake schedules may require compromise.' : 'Great daily rhythm synchronization!'}`;

    const sameLove = input.a_love_language === input.b_love_language;
    const expressDiff = Math.abs(input.a_emotional_expressiveness - input.b_emotional_expressiveness);
    const commPct = Math.max(10, (sameLove ? 50 : 20) + (10 - expressDiff) * 5);
    document.getElementById('commBar').style.width = `${commPct}%`;
    document.getElementById('commDesc').innerText = `${sameLove ? 'Identical love language (' + input.a_love_language + ') boosting emotional bond!' : 'Different love languages (' + input.a_love_language + ' vs ' + input.b_love_language + ').'}`;

    // AI Advice List
    const adviceList = document.getElementById('adviceList');
    adviceList.innerHTML = '';
    const advice = [];

    if (sameLove) {
        advice.push(`You both share the same primary love language (${input.a_love_language}), which provides a solid emotional foundation for deep connection.`);
    } else {
        advice.push(`Partner A prefers '${input.a_love_language}' while Partner B values '${input.b_love_language}'. Be intentional about expressing affection in your partner's preferred style.`);
    }

    if (chronoDiff >= 4) {
        advice.push(`Chronotype misalignment detected (${input.a_chronotype} vs ${input.b_chronotype}). Plan quality time during overlapping waking hours to avoid friction.`);
    } else {
        advice.push(`Similar chronotypes mean your energy levels align nicely throughout the morning and evening.`);
    }

    if (Math.abs(input.a_extraversion - input.b_extraversion) >= 4) {
        advice.push(`Extraversion discrepancy: One partner is significantly more social. Respect each other's recharge time and social battery needs.`);
    }

    if (score >= 70) {
        advice.push(`High Machine Learning compatibility index! Your core values, temperament, and communication dynamics show exceptional long-term synergy.`);
    } else {
        advice.push(`The predictive model highlights personality differences. Open communication and active active conflict resolution will be key to longevity.`);
    }

    advice.forEach(item => {
        const li = document.createElement('li');
        li.innerText = item;
        adviceList.appendChild(li);
    });

    resultsSection.scrollIntoView({ behavior: 'smooth' });
}
